"""
The goal of this module is to determine the position of nodes and edges for an arbitrary
graph. A drop in module for visualization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import NamedTuple, Optional

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


class SugiyamaGraphDrawing:
    def __init__(
        self,
        remove_cycles: Optional[RemoveCycles] = None,
        layer_assignment: Optional[LayerAssignment] = None,
        reuse_nodes: bool = True
    ) -> None:
        self._remove_cycles = RemoveCycleDFS() if remove_cycles is None else remove_cycles
        self._layer_assignment = LayerAssignmentCoffmanGraham(5, 1) \
            if layer_assignment is None \
            else layer_assignment
        self.reuse_nodes = reuse_nodes

    def _extend_layers(
        self, g: Graph, layers: list[list[int]],
        reuse_nodes: bool = False
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
        if reuse_nodes:
            return self._extend_layers_reuse_nodes(g, layers)
        else:
            return self._extend_layers_simple(g, layers)

    def _extend_layers_simple(
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
        self, g: Graph, new_nodes: set[int], edges_ext: Edges, layers_ext: list[list[int]],
        edges_modified: dict[tuple[int, int], list[int]]
    ) -> tuple[set[int], Edges, list[list[int]], dict[tuple[int, int], list[int]]]:
        reversed_edges = Graph(g.vertices | new_nodes, edges_ext).reversed_edges
        layers_index = {v: layer_i
                        for layer_i, vs in enumerate(layers_ext)
                        for v in vs}

        reduced_edges_ext = {n: v.copy() for n, v in edges_ext.items()}

        to_del_nodes: dict[int, int] = {}
        for v in g.vertices:
            ind_v = layers_index[v]
            prev_dummy = {w for w in reversed_edges[v]
                          if w in new_nodes and layers_index[w] == ind_v - 1}
            i = 1
            while prev_dummy:
                new_dummy = prev_dummy.pop()
                for dummy in prev_dummy:
                    if len(reduced_edges_ext[dummy]) == 1:
                        for w in reversed_edges[dummy]:
                            reduced_edges_ext[w].remove(dummy)
                            reduced_edges_ext[w].add(new_dummy)
                        to_del_nodes[dummy] = new_dummy

                i += 1
                prev_dummy = {w for v in prev_dummy
                              for w in reversed_edges[v]
                              if w in new_nodes and layers_index[w] == ind_v - i}

        reduced_layers_ext = [lay[:] for lay in layers_ext]
        for v in to_del_nodes:
            del reduced_edges_ext[v]
            reduced_layers_ext[layers_index[v]].remove(v)

        edges_modified = {k: [to_del_nodes[n] if n in to_del_nodes else n
                              for n in dummy_nodes]
                          for k, dummy_nodes in edges_modified.items()}

        return new_nodes - set(to_del_nodes), reduced_edges_ext, reduced_layers_ext, \
            edges_modified

    def _extend_layers_reuse_nodes(
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
                g, new_nodes, edges_ext, layers_ext, edges_modified_)

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

    def find_pos(
        self,
        graph: Graph,
        inputs: Optional[list[int]] = None,
        outputs: Optional[list[int]] = None,
    ) -> GraphWithPos:
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
        bias = defaultdict(float, {w: -1.0 for v in inputs for w in graph.edges[v]})
        for v in outputs:
            for w in graph.reversed_edges[v]:
                # if w is both in inputs and outputs, it should a bias of 0, otherwise 1
                bias[w] = 0 if w in bias else 1
        layers = self._layer_assignment.assign(
            graph_no_cycles.subgraph(remove=set(inputs) | set(outputs)),
            bias=bias
        )

        # 2b. Adding inputs and outputs, previously ignored
        if inputs:
            layers.insert(0, inputs)
        if outputs:
            layers.append(outputs)
        # 2b. Extending layers to include dummy nodes for long distance edges
        graph_extended, layers_extended, new_nodes, edges_modified = \
            self._extend_layers(graph_no_cycles, layers, reuse_nodes=self.reuse_nodes)

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
