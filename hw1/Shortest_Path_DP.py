from copy import deepcopy


class Shortest_Path_DP():
    """
    Dynamic Programming for Shortest Path Problem
    """

    def __init__(self, graph):
        """
        :param graph: (dic) Acyclic graph contains nodes, arcs and transition costs
        """
        self.graph = graph
        self.stages = []
        self.num_nodes = len(graph)
        self.get_stages()

    def get_stages(self):
        """
        Finds the sorted stages
        """
        graph = deepcopy(self.graph)

        while len(graph) != 0:
            stage_split = []
            current_graph = deepcopy(graph)
            for current_node in current_graph:
                num_visit = 0
                for node in current_graph:
                    if current_node not in current_graph[node].keys():
                        num_visit += 1
                if num_visit == len(current_graph):
                    del graph[current_node]
                    stage_split.append(current_node)

            self.stages.append(stage_split)

    def get_shortest_path(self):
        """
        Finds the shortest path (one optimal path)
        """
        values = {}
        values[self.stages[-1][0]] = 0
        path = [self.stages[-1]]
        del self.stages[-1]
        self.stages.reverse()

        # backward value iteration
        for stage in self.stages:
            max_value = float('inf')
            for node in stage:
                costs = self.graph[node]
                for sub_node in costs:
                    value = costs[sub_node] + values[sub_node]
                    if value <= max_value:
                        max_value = value
                        values[node] = max_value

        # path search based on min values
        for stage in self.stages:
            stage_values = {}
            for node in stage:
                stage_values[node] = values[node]

            min_val = min(stage_values.values())
            res = [k for k, v in stage_values.items() if v == min_val]
            path.append(res)
        path.reverse()

        print("Node Values -", values)
        print("One of the optimal paths -", path)
        print("Optimal traveling time -", max(values.values()))



if __name__ == '__main__':
    graph_1 = {'A': {'B': 1, 'F': 2},
               'B': {'C': 5, 'F': 1},
               'C': {'D': 3, 'H': 4},
               'D': {'H': 1, 'I': 3},
               'F': {'C': 2, 'D': 6, 'G': 3},
               'G': {'D': 1, 'H': 4},
               'H': {'I': 2},
               'I': {},
               }
    print("Graph 1")
    Graph_1 = Shortest_Path_DP(graph_1)
    Graph_1.get_shortest_path()