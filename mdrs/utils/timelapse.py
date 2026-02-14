import re
from datetime import datetime
from typing import Optional


_TS_RE = re.compile(r"(\d{4}\.\d{2}\.\d{2})_(\d{6})")


def parse_timestamp_from_image_name(image_name: str, fallback_date: Optional[str] = None) -> Optional[datetime]:
    """Parse PRMI timestamp.

    Typical: Cotton_T011_L055_2012.08.06_145505_AMC_DPI150.jpg

    Returns datetime, or None if parsing fails.
    """
    m = _TS_RE.search(image_name)
    if m:
        d, t = m.group(1), m.group(2)
        try:
            return datetime.strptime(d + t, "%Y.%m.%d%H%M%S")
        except Exception:
            pass
    if fallback_date:
        try:
            return datetime.strptime(fallback_date, "%Y.%m.%d")
        except Exception:
            return None
    return None


def seq_id(crop: str, location: str, tube_num: str, depth: str, dpi: str) -> str:
    """Canonical time-lapse sequence id for PRMI."""
    return f"{crop}|{location}|{tube_num}|{depth}|{dpi}"


# Backward-compatible alias (older code used this name)
def make_sequence_id(crop: str, location: str, tube_num: str, depth: str, dpi: str) -> str:
    return seq_id(crop, location, tube_num, depth, dpi)


def delta_days(t0: datetime, t1: datetime) -> float:
    return (t1 - t0).total_seconds() / 86400.0
