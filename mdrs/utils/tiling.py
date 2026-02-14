from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Tile:
    y: int
    x: int
    h: int
    w: int


def sliding_tiles(H: int, W: int, tile: int, overlap: int) -> List[Tile]:
    step = max(1, tile - overlap)
    ys = list(range(0, max(1, H - tile + 1), step))
    xs = list(range(0, max(1, W - tile + 1), step))
    if not ys:
        ys = [0]
    if not xs:
        xs = [0]
    if ys[-1] != max(0, H - tile):
        ys.append(max(0, H - tile))
    if xs[-1] != max(0, W - tile):
        xs.append(max(0, W - tile))

    out: List[Tile] = []
    for y in ys:
        for x in xs:
            out.append(Tile(y=y, x=x, h=min(tile, H - y), w=min(tile, W - x)))
    return out


def anchor_params(tile: Tile, H: int, W: int) -> Tuple[float, float, float, float]:
    """Normalized spatial anchor used to keep memory/prompt tokens consistent across tiles."""
    return (
        tile.x / max(1, W),
        tile.y / max(1, H),
        tile.w / max(1, W),
        tile.h / max(1, H),
    )


# Backward-compatible name used by earlier code
anchor_token_params = anchor_params
