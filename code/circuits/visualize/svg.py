from __future__ import annotations

import svgwrite
from svgwrite import px
import math
from math import pi
from typing import Literal, Optional

from .base import CircuitDisplay, Size, Node, Pos, Connection, Angle


Orientation = Literal['left', 'right']


def _offset_for_angle(angle: Optional[Angle], orientation: Orientation
                      ) -> tuple[float, float]:
    if angle is None:
        offset = (0.2, 0.0) if orientation == 'right' else (-0.2, 0.0)
    else:
        align = math.e ** (angle * 1j) * 0.5
        offset = (align.real, -align.imag)
    return offset


def _insert_circuit_in_drawing(
    cd: CircuitDisplay,
    dwg: svgwrite.Drawing,
    zoom: float,
    shift: Pos
) -> None:
    circuit_group = dwg.add(dwg.g(id=cd.name, fill='none', stroke='black', stroke_width=1))
    circuit_group.add(dwg.rect(insert=shift * zoom, size=cd.size * zoom))

    # Defining Nodes/Neurons
    for node in cd.nodes:
        # unit circle (zoomed though)
        center = Pos(node.x + shift.x, node.y + shift.y)
        circuit_group.add(dwg.circle(center=center * zoom, r=zoom/2*px))

    # Defining inputs
    for node in cd.inputs:
        circuit_group.add(dwg.circle(center=(node + shift) * zoom, r=zoom/5*px))

    # Defining outputs
    for node in cd.outputs:
        circuit_group.add(dwg.circle(center=(node + shift) * zoom, r=zoom/5*px,
                                     fill='black'))

    # Defining connections
    for conn in cd.connections:
        start_align = _offset_for_angle(conn.from_angle, orientation='right')
        end_align = _offset_for_angle(conn.to_angle, orientation='left')

        start = (conn.from_ + shift + start_align) * zoom
        finish = (conn.to + shift + end_align) * zoom
        if conn.path is None:
            line = dwg.line(start=start, end=finish)
        else:
            line = dwg.polyline(
                points=[start] + [(p + shift) * zoom for p in conn.path] + [finish]
            )
        line.set_markers((None, None, '#arrow-head'))
        circuit_group.add(line)

    # Inserting more circuits down the line
    for pos, inc_circuit in cd.include:
        _insert_circuit_in_drawing(inc_circuit, dwg, zoom, pos)


def save_svg(cd: CircuitDisplay, path: str, zoom: float = 10) -> None:
    canvas_size = cd.size * zoom
    dwg = svgwrite.Drawing(
        filename=path, size=(canvas_size.x + 1, canvas_size.y + 1),
        viewBox=f'-0.5 -0.5 {canvas_size.x + 1} {canvas_size.y + 1}',
        debug=True)

    # Defining arrow head (marker)
    marker = dwg.marker(insert=(zoom*0.15, zoom*0.15), size=(zoom*0.181, zoom*.3),
                        orient="auto", id='arrow-head')
    # marker.add(dwg.rect(insert=(0, 0), size=(zoom*.18, zoom*.3), fill='gray'))
    marker.add(dwg.polyline(
        points=[(zoom*.025, zoom*.025), (zoom*0.15, zoom*0.15), (zoom*.025, zoom*0.275)],
        stroke='black', fill='none'))
    dwg.defs.add(marker)

    _insert_circuit_in_drawing(cd, dwg, zoom, Pos(0, 0))

    dwg.save()


if __name__ == '__main__':
    simple_circuit = CircuitDisplay(
        "ExampleCircuit",
        Size(10, 10),
        [Node(1, 2), Node(2.5, 2)],
        [Pos(0, 1), Pos(0, 2)],
        [Pos(10, 1), Pos(10, 2)],
        [Connection(Pos(0, 1), Pos(10, 1), None, None),
         Connection(Pos(1, 2), Pos(2.5, 2), .25 * pi, 1 * pi)],
        [(Pos(3, 3),
          CircuitDisplay(
              "SmallerCircuit",
              Size(4, 4),
              [Node(2, 1)],
              [Pos(0, 1)],
              [Pos(4, 1)],
              [Connection(Pos(0, 1), Pos(2, 1), None, .9 * pi, [Pos(1, .5)]),
               Connection(Pos(2, 1), Pos(4, 1), 0, None)],
              []))]
    )

    save_svg(simple_circuit, 'basic_shapes.svg', zoom=20)
