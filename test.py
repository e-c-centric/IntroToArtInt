import heapq #using heapq to implement priority queue for greedy best first search

def greedy_best_first_search(graph, start, goal):
    open_list = []  # Priority queue of nodes to visit
    closed_set = set()  # Set of nodes already visited
    parent_map = {}  # Dictionary to store parent nodes for path reconstruction

    start_node = (start, 0)
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node, _ = heapq.heappop(open_list)
        if current_node == goal:
            path = []
            while current_node != start:
                path.append(current_node)
                current_node = parent_map[current_node]
            path.append(start)
            return path[::-1]
        closed_set.add(current_node)

        for neighbor, edge_weight in graph.get(current_node, []):
            if neighbor not in closed_set:
                heuristic_value = edge_weight #edge weight as heuristic value
                heapq.heappush(open_list, (neighbor, heuristic_value))

                parent_map[neighbor] = current_node

    return None

# Example usage:
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 3), ('C', 2)],
    'C': [('E', 5)],
    'D': [('G', 4), ('F', 2)],
    'E': [('G', 3)],
    'F': [('G', 1)]
}

start_node = 'A'
goal_node = 'G'

path = greedy_best_first_search(graph, start_node, goal_node)
if path:
    print("Path found:", " -> ".join(path))
else:
    print("No path found to the goal.")
