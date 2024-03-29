from __future__ import annotations

import cmath

from typing import NamedTuple, Any, Optional


class Node(NamedTuple):
    x: float
    y: float

    def __mul__(self, other: Any) -> Node:
        if isinstance(other, int):
            return Node(self.x * other, self.y * other)
        return NotImplemented


class Pos(NamedTuple):
    x: float
    y: float

    def __mul__(self, other: Any) -> Pos:
        if isinstance(other, int):
            return Pos(self.x * other, self.y * other)
        return NotImplemented

    def __add__(self, other: Any) -> Pos:
        # This includes `Pos` itself
        if isinstance(other, tuple) and len(other) == 2 \
                and isinstance(other[0], (int, float)) \
                and isinstance(other[1], (int, float)):
            return Pos(self.x + other[0], self.y + other[1])
        return NotImplemented

    def __sub__(self, other: Any) -> Pos:
        # This includes `Pos` itself
        if isinstance(other, tuple) and len(other) == 2 \
                and isinstance(other[0], (int, float)) \
                and isinstance(other[1], (int, float)):
            return Pos(self.x - other[0], self.y - other[1])
        if isinstance(other, (int, float)):
            return Pos(self.x - other, self.y - other)
        return NotImplemented


class Size(NamedTuple):
    x: float
    y: float

    def __add__(self, other: Any) -> Size:
        # This includes `Size` itself
        if isinstance(other, (int, float)):
            return Size(self.x + other, self.y + other)
        if isinstance(other, tuple) and len(other) == 2 \
                and isinstance(other[0], (int, float)) \
                and isinstance(other[1], (int, float)):
            return Size(self.x + other[0], self.y + other[1])
        return NotImplemented

    def __mul__(self, other: Any) -> Size:
        if isinstance(other, int):
            return Size(self.x * other, self.y * other)
        # This includes `Size` itself
        if isinstance(other, tuple) and len(other) == 2 \
                and isinstance(other[0], (int, float)) \
                and isinstance(other[1], (int, float)):
            return Size(self.x + other[0], self.y + other[1])
        return NotImplemented


class AngleRad(NamedTuple):
    """Angle between 0 and 2*pi, and radius (>= 0)"""
    degree: float
    radius: float = 1.0


class Connection(NamedTuple):
    from_: tuple[Pos, AngleRad]
    to: tuple[Pos, AngleRad]
    path: Optional[list[Pos]] = None


class CircuitDisplay(NamedTuple):
    name: Optional[str]
    size: Size
    nodes: dict[int, Node]
    inputs: list[Pos]
    outputs: list[Pos]
    connections: list[Connection]
    # CircuitDisplay is supposed to be a recursive type, but mypy doesn't support it just yet
    # includes: list[tuple[Pos, CircuitDisplay]]
    includes: list[tuple[Pos, Any]]


def straight_line_connection(
    from_: Pos, to: Pos, from_size: float, to_size: float,
    path: Optional[list[Pos]] = None
) -> Connection:
    if path is not None and len(path) == 0:
        path = None
    # first and last coordinates/pos for the given connection path
    first = to if path is None else path[0]
    last = from_ if path is None else path[-1]

    from_angle = AngleRad(-cmath.phase(complex(*(first-from_))), from_size)
    to_angle = AngleRad(-cmath.phase(complex(*(last-to))), to_size)

    return Connection(
        from_=(from_, from_angle),
        to=(to, to_angle),
        path=path
    )
