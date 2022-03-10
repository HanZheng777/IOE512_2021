"IOE 512 HW 3 Code - Han Zheng"


def Bellman_update(graph, actions, para):
    """
    Value iteration via Bellman equations
    :param graph: (dic) a graph contains nodes, arcs and transition costs
    :param actions: (list) action space
    :param para: (str) specify types of Bellman eqs, ie. expectation or optimality
    :return:
    """
    values = {"F": 0}
    policy = {}
    graph_nodes = list(graph)

    for i in range(len(graph), 0, -1):
        current_node = graph_nodes[i - 1]
        costs = list(graph[current_node].values())
        current_sub_node = list(graph[current_node].keys())

        # Find the action under given non-Markovian policy
        if para == "expectation":
            search = current_node
            goes_up = 0
            while search != "S":
                for node in graph:
                    for sub_node in graph[node].keys():
                        action = sub_node.split("_")
                        if search in action:
                            search = node
                            if action[0] == "u":
                                goes_up += 1 # check if has gone up before

            if goes_up > 0 or current_node == "S":
                policy[current_node] = "down"
                values[current_node] = actions[1] * (costs[0] + values[current_sub_node[0].split("_")[1]]) + \
                                       (1 - actions[1]) * (costs[1] + values[current_sub_node[1].split("_")[1]])
            else:
                policy[current_node] = "up"
                values[current_node] = actions[0] * (costs[0] + values[current_sub_node[0].split("_")[1]]) + \
                                       (1 - actions[0]) * (costs[1] + values[current_sub_node[1].split("_")[1]])
        # Greedy policy
        if para == "optimal":

            value_down = actions[1] * (costs[0] + values[current_sub_node[0].split("_")[1]]) + \
                         (1 - actions[1]) * (costs[1] + values[current_sub_node[1].split("_")[1]])
            value_up = actions[0] * (costs[0] + values[current_sub_node[0].split("_")[1]]) + \
                       (1 - actions[0]) * (costs[1] + values[current_sub_node[1].split("_")[1]])

            if value_up < value_down:
                policy[current_node] = "up"
                values[current_node] = value_up
            else:
                policy[current_node] = "down"
                values[current_node] = value_down

    return policy, values


if __name__ == '__main__':
    graph = {'S': {'u_A': 10, 'd_B': 0},
             'A': {'u_C': 500, 'd_D': 0},
             'B': {'u_D': 300, 'd_E': 100},
             'C': {'u_F': 10, 'd_F': 0},
             'D': {'u_F': 1200, 'd_F': 500},
             'E': {'u_F': 20, 'd_F': 20},
             }

    actions = [0.9, 0.4]  # probability of going up for choosing go up or go down

    print("Problem 2")
    policy, values = Bellman_update(graph, actions, "expectation")
    print("Policy:", policy)
    print("Values:", values)

    print("Problem 3")
    policy, values = Bellman_update(graph, actions, "optimal")
    print("Policy:", policy)
    print("Values:", values)
