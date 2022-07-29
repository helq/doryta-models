from __future__ import annotations

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


class Size(NamedTuple):
    x: float
    y: float

    def __mul__(self, other: Any) -> Size:
        if isinstance(other, int):
            return Size(self.x * other, self.y * other)
        # This includes `Size` itself
        if isinstance(other, tuple) and len(other) == 2 \
                and isinstance(other[0], (int, float)) \
                and isinstance(other[1], (int, float)):
            return Size(self.x + other[0], self.y + other[1])
        return NotImplemented


class Angle(NamedTuple):
    """Angle between 0 and 2*pi"""
    degree: float


class Connection(NamedTuple):
    from_: Pos
    to: Pos
    from_angle: Optional[Angle]
    to_angle: Optional[Angle]
    path: Optional[list[Pos]] = None


class CircuitDisplay(NamedTuple):
    name: str
    size: Size
    nodes: list[Node]
    inputs: list[Pos]
    outputs: list[Pos]
    connections: list[Connection]
    include: list[tuple[Pos, CircuitDisplay]]
