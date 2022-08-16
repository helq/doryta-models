"""
The goal of this module is to determine the position of nodes and edges for an arbitrary
graph. A drop in module for visualization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
import statistics
from math import sqrt, ceil
from typing import NamedTuple, Optional, Literal, Iterable, Iterator

from .base import CircuitDisplay, Size, Node, Pos as visPos, straight_line_connection
from .svg import save_svg


Edges = dict[int, set[int]]


class Graph(NamedTuple):
    vertices: set[int]
    edges: dict[int, set[int]]

    def check_correctness(self) -> bool:
        # Edges should be contained in vertices
        if not set(self.edges).issubset(self.vertices):
            return False
        # The end of a vertice should be contained within edges
        for ws in self.edges.values():
            if not ws.issubset(self.vertices):
                return False
        return True

    @property
    def reversed_edges(self) -> dict[int, set[int]]:
        redges: dict[int, set[int]] = {v: set() for v in self.vertices}
        for v, ws in self.edges.items():
            for w in ws:
                assert w in redges
                redges[w].add(v)
        return redges

    def degree(self, v: int) -> int:
        return len(self.edges[v])

    def deep_copy(self) -> Graph:
        return Graph(
            vertices=self.vertices.copy(),
            edges={v: ws.copy() for v, ws in self.edges.items()}
        )

    def subgraph(
        self,
        keep: Optional[set[int]] = None,
        remove: Optional[set[int]] = None
    ) -> Graph:
        if keep is not None and remove is not None:
            raise ValueError("keep and remove cannot be selected at the same time")

        if keep is not None:
            return Graph(
                vertices=self.vertices.intersection(keep),
                edges={v: ws.intersection(keep)
                       for v, ws in self.edges.items()
                       if v in keep}
            )
        elif remove is not None:
            return Graph(
                vertices=self.vertices - remove,
                edges={v: ws - remove
                       for v, ws in self.edges.items()
                       if v not in remove}
            )
        else:
            raise ValueError("Either keep or remove must be assigned")


Pos = tuple[float, float]


class GraphWithPos(NamedTuple):
    width: float
    height: float
    vertices: dict[int, Pos]
    edges: dict[int, list[tuple[int, list[Pos]]]]


class RemoveCycles(ABC):
    @abstractmethod
    def remove_cycles(self, g: Graph) -> tuple[Graph, set[tuple[int, int]]]:
        pass


class RemoveCycleDFS(RemoveCycles):
    """
    Removing cycles using simple DFS
    """
    def __init__(self, reverse: bool = False):
        self.reverse = reverse

    def remove_cycles(self, g: Graph) -> tuple[Graph, set[tuple[int, int]]]:
        if not g.check_correctness():
            raise ValueError("The graph is faulty")
        self._graph = g

        self.removed: set[tuple[int, int]] = set()
        self.on_stack: set[int] = set()
        self.marked: set[int] = set()
        self.edges: dict[int, set[int]] = {}

        for v in self._graph.vertices:
            if v not in self.marked:
                self._dfs(v)
        assert not self.on_stack
        assert len(self._graph.vertices) == len(self.marked)

        return Graph(self._graph.vertices, self.edges), self.removed

    def _dfs(self, v: int) -> None:
        assert v not in self.marked and v not in self.edges

        self.edges[v] = set()
        self.marked.add(v)
        self.on_stack.add(v)

        for w in self._graph.edges[v]:
            if w in self.on_stack:
                self.removed.add((v, w))
                # If this activated, the connection is not deleted, just reversed
                if self.reverse:
                    self.edges[w].add(v)
            else:
                self.edges[v].add(w)

                if w not in self.marked:
                    self._dfs(w)
        self.on_stack.remove(v)


class LayerAssignment(ABC):
    @abstractmethod
    def assign(
        self,
        g: Graph,
        bias: Optional[dict[int, float]] = None
    ) -> list[list[int]]:
        pass


class LayerAssignmentCoffmanGraham(LayerAssignment):
    def __init__(self, w: int, crossings_in_layer: int = 0):
        self.w = w
        self.crossings_in_layer = crossings_in_layer

    def _transitive_reduction(self, g: Graph) -> Edges:
        edges: dict[int, set[int]] = {v: ws.copy() for v, ws in g.edges.items()}
        for v, ws in edges.items():
            remove_from_ws: set[int] = set()

            for w in ws:
                if v == w or w in remove_from_ws:
                    continue
                for x in edges[w]:
                    if x in ws:
                        remove_from_ws.add(x)

            for x in remove_from_ws:
                ws.remove(x)
        return edges

    def _is_vertex_connected_to(self, g: Graph, v: int, vertices: set[int]) -> bool:
        edges = g.edges
        for w in vertices:
            if w in edges[v] or v in edges[w]:
                return True
        return False

    def _vertex_num_conns(self, g: Graph, v: int, vertices: set[int]) -> int:
        edges = g.edges
        num = 0
        for w in vertices:
            if w in edges[v] or v in edges[w]:
                num += 1
        return num

    def _are_restriction_to_add_layer_met(self, g: Graph, v: int, layer: set[int]) -> bool:
        layer_not_full = len(layer) < self.w
        if self.crossings_in_layer == 0:
            return layer_not_full and not self._is_vertex_connected_to(g, v, layer)
        else:
            return layer_not_full \
                and self._vertex_num_conns(g, v, layer) <= self.crossings_in_layer

    def assign(
        self,
        g: Graph,
        bias: dict[int, float] | None = None
    ) -> list[list[int]]:
        if not g.check_correctness():
            raise ValueError("The graph is faulty")
        if bias is None:
            bias = defaultdict(float)

        layers: list[set[int]] = [set()]

        # 0a. Transitive reduction of graph
        edges = self._transitive_reduction(g)
        # 0b. Reversing edges
        reversed_edges = Graph(g.vertices, edges).reversed_edges

        # 1. Sort vertices by the number incoming edges

        # sorted based on the number of incoming edges with an optional bias
        vertices = list(sorted(
            g.vertices,
            key=lambda v: (bias[v], len(reversed_edges[v]))  # type: ignore
        ))

        # 2. Select a vertex at the time such that it uses the highest amount of vertices
        selected_vertices: set[int] = set()
        while vertices:
            for i, v in enumerate(reversed(vertices)):
                if edges[v].issubset(selected_vertices):
                    break
            else:
                raise Exception("The condition to find vertex was never met")
            vertices.pop(-(i+1))

            if self._are_restriction_to_add_layer_met(g, v, layers[-1]):
                layers[-1].add(v)
            else:
                layers.append({v})
            selected_vertices.add(v)

        return [list(lay) for lay in reversed(layers)]


class Llist(Iterable[int]):
    def __init__(self, next: Optional[Llist] = None, value: Optional[int] = None) -> None:
        self.next = next
        self.value = value

    def add_this(self, x: int) -> Llist:
        return Llist(self, x)

    def __iter__(self) -> Iterator[int]:
        node: Optional[Llist] = self
        while node is not None and node.value is not None:
            yield node.value
            node = node.next

    def __len__(self) -> int:
        node: Optional[Llist] = self
        i = 0
        while node is not None and node.value is not None:
            i += 1
            node = node.next
        return i


class SugiyamaGraphDrawing:
    def __init__(
        self,
        remove_cycles: Optional[RemoveCycles] = None,
        layer_assignment: Optional[LayerAssignment] = None,
        reuse_dummy_nodes: bool = True,
        bias_nodes: bool = True,
        vertex_reordering: bool = True
    ) -> None:
        self._remove_cycles = RemoveCycleDFS() if remove_cycles is None else remove_cycles
        self._layer_assignment = LayerAssignmentCoffmanGraham(5, 1) \
            if layer_assignment is None \
            else layer_assignment
        self.reuse_dummy_nodes = reuse_dummy_nodes
        self._bias_nodes = bias_nodes
        self._vertex_reordering = vertex_reordering

    def _extend_dummy_nodes_layers(
        self, g: Graph, layers: list[list[int]],
        reuse_dummy_nodes: bool = False
    ) -> tuple[Graph, list[list[int]], set[int], dict[tuple[int, int], list[int]]]:
        """
        This function takes a graph and its nodes ordered in layers, and adds dummy nodes
        for edges that go across more than one layer. It assumes an acyclic graph as
        input, which the layers take into account so that no edge goes backward.

        This function returns a 4-tuple with the following data:
        1. The new graph
        2. The extended layering assignment
        3. The new nodes
        4. The connections that were removed/replaced in the process (and their
           replacements)
        """
        if reuse_dummy_nodes:
            return self._extend_dummy_nodes_layers_reuse_nodes(g, layers)
        else:
            return self._extend_dummy_nodes_layers_simple(g, layers)

    def _extend_dummy_nodes_layers_simple(
        self, g: Graph, layers: list[list[int]]
    ) -> tuple[Graph, list[list[int]], set[int], dict[tuple[int, int], list[int]]]:
        max_index = max(g.vertices)
        new_index = max_index + 10

        layers_index = {v: layer_i
                        for layer_i, vs in enumerate(layers)
                        for v in vs}

        # Constructing new graph by adding intermediate nodes
        edges_ext = {v: ws.copy() for v, ws in g.edges.items()}
        layers_ext = [list(layer) for layer in layers]
        edges_modified: dict[tuple[int, int], list[int]] = {}
        new_nodes: set[int] = set()
        for v in g.vertices:
            ws_to_del: set[int] = set()
            for w in g.edges[v]:
                # are there intermediate layers between v and w?
                if layers_index[v] + 1 < layers_index[w]:
                    ws_to_del.add(w)
                    edges_ext[v].add(new_index)
                    edges_modified[(v, w)] = []
                # for each layers in between v and w
                for i in range(layers_index[v] + 1, layers_index[w]):
                    # add intermediate nodes and new edges
                    layers_ext[i].append(new_index)
                    edges_ext[new_index] = {w if layers_index[w] == i+1 else new_index + 1}
                    edges_modified[(v, w)].append(new_index)
                    new_nodes.add(new_index)

                    new_index += 1
            edges_ext[v] -= ws_to_del

        return Graph(g.vertices | new_nodes, edges_ext), layers_ext, new_nodes, edges_modified

    def _combine_dummy_nodes_backwards(
        self, vertices: set[int], new_nodes: set[int],
        edges_ext: Edges, layers_ext: list[list[int]],
        edges_modified: dict[tuple[int, int], list[int]]
    ) -> tuple[set[int], Edges, list[list[int]], dict[tuple[int, int], list[int]]]:
        reversed_edges = Graph(vertices | new_nodes, edges_ext).reversed_edges
        layers_index = {v: layer_i
                        for layer_i, vs in enumerate(layers_ext)
                        for v in vs}

        reduced_edges_ext = {n: v.copy() for n, v in edges_ext.items()}

        to_del_nodes: dict[int, int] = {}
        for v in vertices:
            ind_v = layers_index[v]
            prev_dummy = {w for w in reversed_edges[v]
                          if (w in new_nodes and layers_index[w] == ind_v - 1
                              and len(reduced_edges_ext[w]) == 1)}
            i = 1
            while prev_dummy:
                new_dummy = prev_dummy.pop()
                for dummy in prev_dummy:
                    for w in reversed_edges[dummy]:
                        reduced_edges_ext[w].remove(dummy)
                        reduced_edges_ext[w].add(new_dummy)
                    to_del_nodes[dummy] = new_dummy

                i += 1
                prev_dummy = {w for v in prev_dummy
                              for w in reversed_edges[v]
                              if (w in new_nodes and layers_index[w] == ind_v - i
                                  and len(reduced_edges_ext[w]) == 1)}

        reduced_layers_ext = [lay[:] for lay in layers_ext]
        for v in to_del_nodes:
            del reduced_edges_ext[v]
            reduced_layers_ext[layers_index[v]].remove(v)

        edges_modified = {k: [to_del_nodes[n] if n in to_del_nodes else n
                              for n in dummy_nodes]
                          for k, dummy_nodes in edges_modified.items()}

        return new_nodes - set(to_del_nodes), reduced_edges_ext, reduced_layers_ext, \
            edges_modified

    def _extend_dummy_nodes_layers_reuse_nodes(
        self, g: Graph, layers: list[list[int]]
    ) -> tuple[Graph, list[list[int]], set[int], dict[tuple[int, int], list[int]]]:
        max_index = max(g.vertices)
        new_index = max_index + 10

        layers_index = {v: layer_i
                        for layer_i, vs in enumerate(layers)
                        for v in vs}

        # Constructing new graph by adding intermediate nodes
        edges_ext = {v: ws.copy() for v, ws in g.edges.items()}
        layers_ext = [list(layer) for layer in layers]
        edges_modified: dict[int, tuple[list[int], dict[int, int]]] = \
            defaultdict(lambda: ([], defaultdict(int)))
        new_nodes: set[int] = set()
        for v in g.vertices:
            ws_to_del: set[int] = set()
            for w in g.edges[v]:
                start_new_nodes = layers_index[v] + 1
                w_pos = layers_index[w]
                # are there intermediate layers between v and w?
                if start_new_nodes >= w_pos:
                    continue
                ws_to_del.add(w)
                nodes_ahead = len(edges_modified[v][0])
                # have nodes already been created between v and w
                if edges_modified[v][0]:
                    if start_new_nodes + nodes_ahead < w_pos:
                        # connect last node in new change of nodes to new_index node
                        last_node_in_line = edges_modified[v][0][-1]
                        edges_ext[last_node_in_line].add(new_index)
                        edges_modified[v][1][w] = nodes_ahead
                    else:
                        # make a connection from an intermediate node to w
                        intermediate_nodes = w_pos - start_new_nodes
                        node_before_w = edges_modified[v][0][intermediate_nodes - 1]
                        edges_ext[node_before_w].add(w)
                        edges_modified[v][1][w] = intermediate_nodes
                else:
                    edges_ext[v].add(new_index)
                # for each of the remaining layers in between v and w, add new nodes
                for i in range(start_new_nodes + nodes_ahead, w_pos):
                    # add intermediate nodes and new edges
                    layers_ext[i].append(new_index)
                    edges_ext[new_index] = {w if w_pos == i+1 else new_index + 1}
                    edges_modified[v][0].append(new_index)
                    edges_modified[v][1][w] += 1
                    new_nodes.add(new_index)

                    new_index += 1
            edges_ext[v] -= ws_to_del

        edges_modified_ = {(v, w): ls[:size] for v, (ls, ws) in edges_modified.items()
                           for w, size in ws.items()}

        # This call can be commented and the code will keep working. This call fuses
        # redundant dummy nodes backwards (starting at some node and going backwards)
        new_nodes, edges_ext, layers_ext, edges_modified_ = \
            self._combine_dummy_nodes_backwards(
                g.vertices, new_nodes, edges_ext, layers_ext, edges_modified_)

        return Graph(g.vertices | new_nodes, edges_ext), \
            layers_ext, new_nodes, edges_modified_

    def _graph_with_pos_from_layers_extended(
        self,
        graph: Graph,
        layers: list[list[int]],
        new_nodes: set[int],
        edges_modified: dict[tuple[int, int], list[int]]
    ) -> GraphWithPos:
        width = len(layers)
        height = max(len(lay) for lay in layers)

        layers_index = {v: (layer_i, j)
                        for layer_i, vs in enumerate(layers)
                        for j, v in enumerate(vs)}
        assert (graph.vertices | new_nodes) == set(layers_index)

        def _find_path_edge(v: int, w: int) -> list[tuple[float, float]]:
            if (v, w) in edges_modified:
                return [layers_index[x] for x in edges_modified[(v, w)]]
            return []

        return GraphWithPos(
            width=width,
            height=height,
            vertices={
                v: (i, j)
                for i, layer in enumerate(layers)
                for j, v in enumerate(layer)
                if v not in new_nodes
            },
            edges={v: [(w, _find_path_edge(v, w)) for w in ws]
                   for v, ws in graph.edges.items()}
        )

    def _connected_nodes_in_list(self, graph: Graph, nodes: list[int]) -> dict[int, set[int]]:
        this_nodes = set(nodes)
        edges = graph.edges
        reversed_edges = graph.reversed_edges
        groups = {}
        for n in nodes:
            groups[n] = edges[n].intersection(this_nodes) \
                | reversed_edges[n].intersection(this_nodes)
            groups[n].add(n)
        return groups

    def _sort_layers_median(
        self,
        graph: Graph,
        layers: list[list[int]],
        direction: Literal[-1, 1],  # negative or positive direction
        include_last: bool
    ) -> list[list[int]]:
        layers_ordered = [lay.copy() for lay in layers]
        edges_prev = graph.edges if direction == 1 else graph.reversed_edges

        start = 1 if direction == 1 else len(layers) - 2
        end = len(layers) if direction == 1 else -1

        prev_lay = layers_ordered[0] if direction == 1 else layers_ordered[-1]
        if not include_last:
            end = -direction
        for lay in layers_ordered[start:end:direction]:
            keys: dict[int, float] = {}

            # Computing median of previous positions
            for j, v in enumerate(lay):
                prev_conn_pos = [i for i, w in enumerate(prev_lay) if v in edges_prev[w]]
                if prev_conn_pos:
                    keys[v] = statistics.median(prev_conn_pos)
                else:
                    keys[v] = j

            # Checking if at least two nodes are connected to each other within the layer
            groups = self._connected_nodes_in_list(graph, lay)
            internal_connections = any(len(group) > 1 for group in groups.values())
            if internal_connections:
                new_keys = {}
                pos = {v: i for i, v in enumerate(lay)}
                for v, ws_in_group in groups.items():
                    average = statistics.mean(keys[w] for w in ws_in_group)
                    average_pos = statistics.mean(pos[w] for w in ws_in_group)
                    new_keys[v] = (average, average_pos, keys[v])
                keys = new_keys  # type: ignore

            lay.sort(key=keys.__getitem__)
            prev_lay = lay

        return layers_ordered

    def _shufle_layers(
        self,
        layers: list[list[int]],
        inputs: bool,
        outputs: bool
    ) -> list[list[int]]:
        from random import shuffle
        layers = [lay.copy() for lay in layers]
        start = 1 if inputs else 0
        end = len(layers) - 2 if outputs else len(layers) - 1
        for lay in layers[start:end]:
            shuffle(lay)
        return layers

    def _vertex_ordering_stage(
        self,
        graph: Graph,
        layers: list[list[int]],
        inputs: bool,
        outputs: bool
    ) -> list[list[int]]:
        # layers = self._shufle_layers(layers, inputs, outputs)
        for i in range(3):
            # i = 0
            layers = self._sort_layers_median(
                graph, layers,
                direction=1 if i % 2 else -1,
                include_last=not (outputs if i % 2 else inputs))
        return layers

    def _find_edges_modified_in_new_graph(
        self,
        graph: Graph,
        graph_original: Graph
    ) -> dict[tuple[int, int], list[int]]:
        edges_modified: dict[tuple[int, int], list[int]] = {}

        def dfs_helper(w: int, path: Llist) -> None:
            # in case we have reached a non-dummy node
            if w in graph_original.vertices:
                if len(path):
                    pathlist = list(path)
                    pathlist.reverse()
                    edges_modified[(v, w)] = pathlist
                return

            # general case
            new_path = path.add_this(w)
            for k in graph.edges[w]:
                dfs_helper(k, new_path)

        for v, ws in graph_original.edges.items():
            for w in ws:
                if w not in graph.edges[v] and (v, w) not in edges_modified:
                    for k in graph.edges[v]:
                        dfs_helper(k, Llist())
                    assert (v, w) in edges_modified

        return edges_modified

    def _find_layers_index(self, layers: list[list[int]]) -> dict[int, tuple[int, int]]:
        return {v: (layer_i, j)
                for layer_i, vs in enumerate(layers)
                for j, v in enumerate(vs)}

    def _extra_layers_for_ease_of_visualization(
        self,
        graph: Graph,
        graph_original: Graph,
        layers: list[list[int]],
        new_nodes: set[int],
    ) -> tuple[Graph, list[list[int]], set[int], dict[tuple[int, int], list[int]]]:
        new_graph = graph.deep_copy()
        new_layers = [lay.copy() for lay in layers]
        new2_nodes = new_nodes.copy()

        new_index = max(graph.vertices) + 10
        layers_index = self._find_layers_index(layers)

        vertices_to_check = graph.vertices
        while vertices_to_check:
            v = vertices_to_check.pop()
            if new_graph.degree(v) > 3:
                num_subnodes = min(int(ceil(sqrt(new_graph.degree(v)))), 3)
                lay_i, v_pos = layers_index[v]
                new_lay: list[int] = []
                new_layers.insert(lay_i + 1, new_lay)
                for j, w in enumerate(new_layers[lay_i]):
                    # Finding where each new node should point to
                    if j == v_pos:
                        to_add = num_subnodes
                        ws_as_list = list(new_graph.edges[w])
                        ws_as_list.sort(key=new_layers[lay_i + 2].index)
                        per_chunk = int(ceil(len(ws_as_list) / num_subnodes))
                        chunks = [set(ws_as_list[per_chunk*k:per_chunk*(k+1)])
                                  for k in range(num_subnodes)]
                    else:
                        to_add = 1
                        chunks = [new_graph.edges[w].copy()]

                    # Creating new nodes for layer
                    added: set[int] = set()
                    for k in range(to_add):
                        new_graph.vertices.add(new_index)
                        new_graph.edges[new_index] = chunks[k]
                        new_lay.append(new_index)
                        added.add(new_index)

                        new_index += 1
                    new_graph.edges[w] = added

                new2_nodes |= set(new_lay)
                vertices_to_check |= set(new_lay)
                layers_index |= self._find_layers_index(new_layers)

        return new_graph, new_layers, new2_nodes, \
            self._find_edges_modified_in_new_graph(new_graph, graph_original)

    def _check_for_possible_inconsistencies(self) -> None:
        if self.reuse_dummy_nodes and isinstance(self._remove_cycles, RemoveCycleDFS) \
                and self._remove_cycles.reverse:
            raise Exception("If you are reusing dummy nodes, it's not a good idea to allow "
                            "discarded egdes in reverse")

    def find_pos(
        self,
        graph: Graph,
        inputs: Optional[list[int]] = None,
        outputs: Optional[list[int]] = None,
    ) -> GraphWithPos:
        self._check_for_possible_inconsistencies()

        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = []
        # Sugiyama method steps
        # 1. Removing cycles
        graph_no_cycles, _ = self._remove_cycles.remove_cycles(graph)

        # 2. Layering assignments

        # If input or output nodes are defined, they will be ignored
        # A positive bias will nudge the vertex to the last layers, a negative bias to the
        # first layers. Nodes connected to input nodes should be closer to the input. The
        # same goes for output nodes
        if self._bias_nodes:
            bias = defaultdict(float, {w: -1.0 for v in inputs for w in graph.edges[v]})
            for v in outputs:
                for w in graph.reversed_edges[v]:
                    # if w is both in inputs and outputs, it should a bias of 0, otherwise 1
                    bias[w] = 0 if w in bias else 1
        else:
            bias = None

        layers = self._layer_assignment.assign(
            graph_no_cycles.subgraph(remove=set(inputs) | set(outputs)),
            bias=bias
        )

        # 2b. Adding inputs and outputs, previously ignored
        if inputs:
            layers.insert(0, inputs)
        if outputs:
            layers.append(outputs)
        # 2b. Adding dummy nodes for long distance edges
        graph_extended, layers_extended, new_nodes, edges_modified = \
            self._extend_dummy_nodes_layers(graph_no_cycles, layers,
                                            reuse_dummy_nodes=self.reuse_dummy_nodes)

        # 3. Vertex ordering
        if self._vertex_reordering:
            layers_extended = self._vertex_ordering_stage(
                graph_extended, layers_extended, inputs=bool(inputs), outputs=bool(outputs))
        # Intermediate step. Adding extra layers for nodes with too many connections
        graph_extended, layers_extended, new_nodes, edges_modified = \
            self._extra_layers_for_ease_of_visualization(
                graph_extended, graph_no_cycles, layers_extended, new_nodes)
        # new_nodes, edges_extended, layers_extended, edges_modified = \
        #     self._combine_dummy_nodes_backwards(
        #         graph_extended.vertices - new_nodes, new_nodes, graph_extended.edges,
        #         layers_extended, edges_modified)
        # Repeating ordering after adding new nodes
        if self._vertex_reordering:
            layers_extended = self._vertex_ordering_stage(
                graph_extended, layers_extended, inputs=bool(inputs), outputs=bool(outputs))

        # Determining positions vertices and edges from graph data
        graph_with_pos = self._graph_with_pos_from_layers_extended(
            graph_no_cycles, layers_extended, new_nodes, edges_modified)

        # Extra. Reversing reversed/deleted connections
        # TODO: Implement this stage!!

        return graph_with_pos


def graph_to_circuitdisplay(g: GraphWithPos) -> CircuitDisplay:
    nodes = g.vertices
    index = {v: i for i, v in enumerate(g.vertices)}

    nodes_ = {index[v]: Node(n[0] * 1.5 + 1, n[1] * 1.5 + 1) for v, n in nodes.items()}

    return CircuitDisplay(
        name=None,
        size=Size(g.width * 1.5 + .5, g.height * 1.5 + .5),
        nodes=nodes_,
        inputs=[],
        outputs=[],
        connections=[
            straight_line_connection(
                visPos(*nodes_[index[v]]),
                visPos(*nodes_[index[w]]),
                0.5, 0.5,
                path=[visPos(pos[0] * 1.5 + 1, pos[1] * 1.5 + 1) for pos in path]
            )
            for v in g.vertices
            for w, path in g.edges[v]
        ],
        includes=[]
    )


if __name__ == '__main__':
    test_a = Graph({1, 2, 3, 4, 5, 6},
                   {1: {2, 3}, 2: {4}, 3: set(), 4: {5, 3}, 5: {2, 3}, 6: set()})

if False and __name__ == '__main__':
    print("Original graph:")
    print(test_a)

    print("\nGraph with cycles removed:")
    test_a_rem, rmd = RemoveCycleDFS().remove_cycles(test_a)
    print(test_a_rem)
    print("Edges removed/reversed:", rmd)

    print("\nNodes order")
    assignt = LayerAssignmentCoffmanGraham(5, 1).assign(test_a_rem)
    print(assignt)

if True and __name__ == '__main__':
    test_a_pretty = SugiyamaGraphDrawing(
        remove_cycles=RemoveCycleDFS(reverse=True),
        layer_assignment=LayerAssignmentCoffmanGraham(w=2, crossings_in_layer=1)
    ).find_pos(test_a)
    print(test_a_pretty)
    test_a_cd = graph_to_circuitdisplay(test_a_pretty)

    save_svg(test_a_cd, 'testing_automatic_visualization.svg', zoom=30)
