# utils/parsing.py
# Pure helpers dùng chung cho api_server và vector_store (stdlib-only).
import re
from typing import Any, Dict, Optional

_FRAME_RE = re.compile(r"__frame_(\d+)")


def parse_hms_to_seconds(s: str) -> Optional[float]:
    """Parse 'HH:MM:SS[.fff]' hoặc 'MM:SS[.fff]' -> seconds. Trả None nếu không parse được."""
    try:
        parts = s.strip().split(":")
        if len(parts) == 3:
            h, m, sec = int(parts[0]), int(parts[1]), float(parts[2])
            return h * 3600 + m * 60 + sec
        if len(parts) == 2:
            m, sec = int(parts[0]), float(parts[1])
            return m * 60 + sec
    except (AttributeError, ValueError):
        pass
    return None


def extract_frame_id_from_payload(payload: Optional[Dict[str, Any]]) -> Optional[int]:
    """Lấy frame index từ payload: ưu tiên frame.index, fallback parse '__frame_000123' trong filename/file_path."""
    payload = payload or {}
    frame = payload.get("frame") or {}
    if isinstance(frame, dict) and frame.get("index") is not None:
        try:
            return int(frame["index"])
        except (TypeError, ValueError):
            pass
    for key in ("filename", "file_path"):
        m = _FRAME_RE.search(payload.get(key) or "")
        if m:
            return int(m.group(1))
    return None
