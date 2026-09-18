"""
In-memory session store.
Maps session_id (str) -> list of DatasetRecord dicts.

Each DatasetRecord contains:
  - name: original filename
  - df: pandas DataFrame (normalized)
  - original_columns: mapping from normalized -> original column names
  - metadata: {rows, columns, dtypes, sample}
"""

import threading
from typing import Any

_store: dict[str, list[dict[str, Any]]] = {}
_lock = threading.Lock()


def get_datasets(session_id: str) -> list[dict[str, Any]]:
    """Return all datasets for a session (empty list if none)."""
    with _lock:
        return _store.get(session_id, [])


def add_dataset(session_id: str, record: dict[str, Any]) -> None:
    """Append a dataset record to the session."""
    with _lock:
        if session_id not in _store:
            _store[session_id] = []
        _store[session_id].append(record)


def remove_dataset(session_id: str, filename: str) -> bool:
    """Remove a dataset by filename. Returns True if removed."""
    with _lock:
        if session_id not in _store:
            return False
        original_len = len(_store[session_id])
        _store[session_id] = [
            d for d in _store[session_id] if d["name"] != filename
        ]
        return len(_store[session_id]) < original_len


def clear_session(session_id: str) -> None:
    """Remove all datasets for a session."""
    with _lock:
        _store.pop(session_id, None)


def list_sessions() -> list[str]:
    """Return all active session IDs (for debugging)."""
    with _lock:
        return list(_store.keys())
