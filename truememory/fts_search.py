""" uses BM25 algorithm to search for a query in a text file and returns the line number and the line containing the query.
it internally uses TFIDF
returns the raw negative values so we need to uses normalization 
 """

from __future__ import annotations  
# Allows us to use modern Python type hints

import sqlite3
from typing import Any

from truememory.storage import _deserialize_metadata, directive_filter_sql, select_message_cols


def _build_safe_query(query: str) -> str:
    """Convert natural language query into a safe FTS5 MATCH string.

    Prevents syntax injection by double-quoting each word token and joining with OR.
    Example: "user likes python" -> '"user" OR "likes" OR "python"'

    chr(34) is the ASCII code for "
    fts5 has it own query language and so we combine them with OR
    SQLite FTS5 has its own query language with special operators: AND, OR, NOT, *, :, "", and ().

    f'"{w}"': Wraps each individual word in quotes ('"hello"', '"world"'). In SQLite FTS5, wrapping a word in double quotes forces FTS5 to treat it as a literal plain text string, disabling any special syntax characters inside it!

    """
    tokens = [f'"{w.replace(chr(34), "")}"' for w in query.split() if w.strip()]
    return " OR ".join(tokens) if tokens else '""'


def _normalize_scores(results: list[dict[str, Any]]) -> None:
    """Normalize raw BM25 scores (negative values) to a 0.0 - 1.0 range in place.

    FTS5 rank values are negative (more negative = higher relevance).
    We scale them so the top result gets 1.0 and lower matches scale down towards 0.0.
    """
    if not results:
        return

    min_score = min(r["raw_score"] for r in results)
    max_score = max(r["raw_score"] for r in results)
    score_range = max_score - min_score

    if score_range == 0:
        for r in results:
            r["score"] = 1.0
    else:
        for r in results:
            r["score"] = (max_score - r["raw_score"]) / score_range

# ---------------------------------------------------------------------------
# Internal Row Converter Helper
# ---------------------------------------------------------------------------

def _rows_to_results(rows: list[Any]) -> list[dict[str, Any]]:
    """Convert raw SQLite SQL query rows into clean Python result dictionaries."""
    results = []
    for row in rows:
        r = dict(row)
        # Parse JSON metadata safely
        r["metadata"] = _deserialize_metadata(r.get("metadata"))
        r["directive"] = bool(r.get("directive", 0))
        # Store raw BM25 score under 'raw_score', initialize 'score' to 0.0
        r["raw_score"] = r.pop("bm25_score", 0.0)
        r["score"] = 0.0
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Public Full-Text Search API
# ---------------------------------------------------------------------------

def search_fts(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
    include_directives: bool = False,
) -> list[dict[str, Any]]:
    """Search stored memories using SQLite FTS5 with BM25 ranking and normalized scores.

    Args:
        conn: Open SQLite database connection.
        query: Search query string (plain text or keywords).
        limit: Maximum number of search results to return.
        include_directives: If False (default), excludes directive=1 instructions.

    Returns:
        List of memory dicts ordered by relevance (best match first), with normalized score (0.0-1.0).
    """
    if not query or not query.strip():
        return []

    # 1. Sanitize user query string
    safe_query = _build_safe_query(query)
    if safe_query == '""':
        return []

    # 2. Build directive filter clause
    dir_filter = directive_filter_sql(conn, alias="m", include_directives=include_directives)

    # 3. Construct FTS5 SQL JOIN query
    sql = f"""
    SELECT m.id, m.content, m.sender, m.recipient, m.timestamp,
           m.category, m.modality, m.directive, m.metadata,
           messages_fts.rank AS bm25_score
    FROM messages_fts
    JOIN messages m ON m.id = messages_fts.rowid
    WHERE messages_fts MATCH ?{dir_filter}
    ORDER BY messages_fts.rank ASC
    LIMIT ?;
    """

    try:
        rows = conn.execute(sql, (safe_query, limit)).fetchall()
    except sqlite3.OperationalError:
        return []

    # 4. Convert rows to dictionary results and normalize scores
    results = _rows_to_results(rows)
    _normalize_scores(results)

    return results