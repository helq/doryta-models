from __future__ import annotations

import svgwrite
from svgwrite import px
import math
from math import pi, ceil, floor
from typing import Literal
from pathlib import Path

from .base import CircuitDisplay, Size, Node, Pos, Connection, AngleRad


Orientation = Literal['left', 'right']


def _offset_for_angle(angle: AngleRad) -> tuple[float, float]:
    align = math.e ** (angle.degree * 1j) * angle.radius
    return (align.real, -align.imag)


def _insert_circuit_in_drawing(
    cd: CircuitDisplay,
    dwg: svgwrite.Drawing,
    zoom: float,
    print_dummy: bool,
    shift: Pos
) -> None:
    id_param = {} if cd.name is None else {'id': cd.name}
    # Inserting more circuits down the line
    for pos, inc_circuit in cd.includes:
        _insert_circuit_in_drawing(inc_circuit, dwg, zoom, print_dummy, shift + pos)

    circuit_group = dwg.add(dwg.g(fill='none', stroke='black', stroke_width=1, **id_param))
    circuit_group.add(dwg.rect(insert=shift * zoom, size=cd.size * zoom))

    # Defining Nodes/Neurons
    for node in cd.nodes.values():
        # unit circle (zoomed though)
        center = Pos(node.x + shift.x, node.y + shift.y)
        circuit_group.add(dwg.circle(center=center * zoom, r=zoom/2*px))

    # Defining inputs
    for pos in cd.inputs:
        circuit_group.add(dwg.circle(center=(pos + shift) * zoom, r=zoom/8*px))

    # Defining outputs
    for pos in cd.outputs:
        circuit_group.add(dwg.circle(center=(pos + shift) * zoom, r=zoom/8*px,
                                     fill='black'))

    # Defining connections
    dummy_nodes: set[Pos] = set()
    for conn in cd.connections:
        start_align = _offset_for_angle(conn.from_[1])
        end_align = _offset_for_angle(conn.to[1])

        start = (conn.from_[0] + shift + start_align) * zoom
        finish = (conn.to[0] + shift + end_align) * zoom
        if conn.path is None:
            # This is an awful trick to make intersections clearer.
            # Gotta find a prettier way!
            line2 = dwg.line(start=start, end=finish, stroke='white', stroke_width=5)
            line = dwg.line(start=start, end=finish)
        else:
            in_between_positions = [(p + shift) * zoom for p in conn.path]
            points = [start] + in_between_positions + [finish]
            line2 = dwg.polyline(points=points, stroke='white', stroke_width=3)
            line = dwg.polyline(points=points)
            dummy_nodes |= set(in_between_positions)
        line.set_markers((None, None, '#arrow-head'))
        circuit_group.add(line2)
        circuit_group.add(line)

    if print_dummy:
        for pos in dummy_nodes:
            circuit_group.add(dwg.rect(
                insert=pos - 0.06 * zoom, size=(zoom*.12, zoom*.12),
                fill='white'
            ))


def _add_grid(dwg: svgwrite.Drawing, size: Size, zoom: float) -> None:
    grid_group = dwg.add(dwg.g(fill='none', stroke='#ccc', stroke_width=1))
    hzoom = zoom / 2  # half zoom
    dsize_x, dsize_y = size * 2
    width = int(dsize_x) + 1 if ceil(dsize_x) == floor(dsize_x) else int(dsize_x)
    height = int(dsize_y) + 1 if ceil(dsize_y) == floor(dsize_y) else int(dsize_y)

    # vertical lines
    for i in range(width):
        ver_line = dwg.line(start=(i * hzoom, -.06 * zoom),
                            end=(i * hzoom, size.y * zoom))
        if i % 2 == 0:
            ver_line.dasharray([4, 2])
        else:
            ver_line.dasharray([1, 5])
        grid_group.add(ver_line)
        if i % 2 == 0:
            head_text = svgwrite.text.Text(
                str(i//2), insert=(i*hzoom, -.2 * zoom), fill='#ccc',
                text_anchor='middle', dominant_baseline='middle')
        grid_group.add(head_text)

    # horizontal lines
    for i in range(height):
        hor_line = dwg.line(start=(-.06 * zoom, i * hzoom),
                            end=(size.x * zoom, i * hzoom))
        if i % 2 == 0:
            hor_line.dasharray([4, 2])
        else:
            hor_line.dasharray([1, 5])
        grid_group.add(hor_line)
        if i and i % 2 == 0:
            head_text = svgwrite.text.Text(
                str(i//2), insert=(-.2 * zoom, i*hzoom), fill='#ccc',
                text_anchor='middle', dominant_baseline='middle')
            grid_group.add(head_text)


def save_svg(
    cd: CircuitDisplay,
    path: str | Path,
    zoom: float = 10,
    grid: bool = False,
    print_dummy: bool = False
) -> None:
    grid_offset = .5 * zoom if grid else 0
    canvas_size = cd.size * zoom + grid_offset

    dwg = svgwrite.Drawing(
        filename=path,
        size=(canvas_size.x + 1, canvas_size.y + 1),
        viewBox=f'{-grid_offset - .5} {-grid_offset - .5} '
                f'{canvas_size.x + 1} {canvas_size.y + 1}',
        debug=True)

    # Defining arrow head (marker)
    marker = dwg.marker(insert=(zoom*0.12, zoom*0.10), size=(zoom*0.121, zoom*.2),
                        orient="auto", id='arrow-head')
    # marker.add(dwg.rect(insert=(0, 0), size=(zoom*.141, zoom*.2), fill='gray'))
    marker.add(dwg.polyline(
        points=[(zoom*.025, zoom*.025), (zoom*0.10, zoom*0.10), (zoom*.025, zoom*0.175)],
        stroke='black', fill='none'))
    dwg.defs.add(marker)

    # Making grid
    if grid:
        _add_grid(dwg, cd.size, zoom)

    _insert_circuit_in_drawing(cd, dwg, zoom, print_dummy, Pos(0, 0))

    dwg.save()


if __name__ == '__main__':
    simple_circuit = CircuitDisplay(
        "ExampleCircuit",
        Size(10, 10),
        {0: Node(1, 2), 1: Node(2.5, 2)},
        [Pos(0, 1), Pos(0, 2)],
        [Pos(10, 1), Pos(10, 2)],
        [Connection((Pos(0, 1), AngleRad(0, .125)), (Pos(10, 1), AngleRad(pi, .125))),
         Connection((Pos(1, 2), AngleRad(.25 * pi)), (Pos(2.5, 2), AngleRad(pi)))],
        [(Pos(3, 3),
          CircuitDisplay(
              "SmallerCircuit",
              Size(4, 4),
              {0: Node(2, 1)},
              [Pos(0, 1)],
              [Pos(4, 1)],
              [Connection((Pos(0, 1), AngleRad(0, .125)),
                          (Pos(2, 1), AngleRad(.9 * pi)),
                          [Pos(1, .5)]),
               Connection((Pos(2, 1), AngleRad(0)), (Pos(4, 1), AngleRad(pi, .125)))],
              []))]
    )

    save_svg(simple_circuit, 'basic_shapes.svg', zoom=20)
