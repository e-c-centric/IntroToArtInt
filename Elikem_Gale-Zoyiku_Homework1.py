import heapq #using heapq to access to the heappush and heappop functions for priority queue for greedy best first search
import random #using random to shuffle the list of neighbors for depth first search
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 3), ('C', 2)],
    'C': [('E', 5)],
    'D': [('G', 4),('F', 2)],
    'E': [('G', 3)],
    'F': [('G', 1)]
}

heuristics = {
    'A': 5,
    'B': random.randint(0,6),
    'C': 4,
    'D': 3,
    'E': 3,
    'F': 1,
    'G': 0
}

start = 'A'
goal = 'G'

def depth_first_search(graph, start, goal):
    stack = []  # Stack to store nodes to be explored
    visited = set()  # Set to store visited nodes
    parent_map = {}  # Dictionary to store parent nodes for path reconstruction
    stack.append(start)
    while stack:
        current_node = stack.pop()
        if current_node == goal:
            path = []
            while current_node != start:
                path.append(current_node)
                current_node = parent_map[current_node]
            path.append(start)
            return path[::-1]
        # Mark the current node as visited
        visited.add(current_node)
        # Explore neighbors of the current node
        neighbors = graph.get(current_node, [])
        random.shuffle(neighbors)  # Shuffle the list of neighbors
        for neighbor, _ in neighbors:
            if neighbor not in visited and neighbor not in stack:
                stack.append(neighbor)
                # Update the parent node for path reconstruction
                parent_map[neighbor] = current_node
    return None

print("Depth First Search")
path = depth_first_search(graph, start, goal)
if path:
    print("Path found:", " -> ".join(path))
else:
    print("No path found to the goal.")



def all_paths(graph, start, goal, path=[]):
    '''Returns all paths from start to goal in the graph.Answers Question 2'''
    path = path + [start]
    if start == goal:
        return [path]
    paths = []
    for neighbor, _ in graph[start]:
        if neighbor not in path:
            new_paths = all_paths(graph, neighbor, goal, path)
            for p in new_paths:
                paths.append(p)
    return paths

print("\nAll Paths")
paths = all_paths(graph, start, goal)
if paths:
    print("Paths found:")
    n=1
    for path in paths:
        print("Path " + str(n) +". " + " -> ".join(path))
        n+=1
else:
    print("No paths found to the goal.")


def greedy_best_first_search(graph, start, goal):
    '''Greedy best-first search algorithm answering Question 3'''
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

print("\nGreedy Best First Search")
path = greedy_best_first_search(graph, start, goal)
if path:
    print("Path found:", " -> ".join(path))
else:
    print("No path found to the goal.")


def astar(graph, start, goal, heuristics):
    '''A* algorithm answering Question 4'''
    visited = set()
    queue = [(heuristics[start], 0, [start])]
    while queue:
        (h, cost, path) = queue.pop(0)
        node = path[-1]
        if node not in visited:
            if node == goal:
                return path
            visited.add(node)
            for neighbor, weight in graph[node]:
                if neighbor not in visited:
                    g = cost + weight
                    h = heuristics[neighbor]
                    f = g + h
                    queue.append((f, g, path + [neighbor]))
                    queue.sort()
    return None

print("\nA* Search")
path = astar(graph, start, goal,heuristics)
if path:
    print("Path found:", " -> ".join(path))
else:
    print("No path found to the goal.")