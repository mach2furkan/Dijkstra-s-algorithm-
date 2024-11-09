import heapq
import time

class DijkstraAlgorithm:
    """
    A class to represent Dijkstra's shortest path algorithm for a weighted graph.
    Provides functionalities for calculating shortest paths, managing edges, and graph analysis.
    """

    def __init__(self, graph):
        """
        Initialize the algorithm with a graph.
        :param graph: Dictionary where each node points to another dictionary of neighboring nodes and weights.
        """
        self.graph = graph
        self.validate_graph()

    def validate_graph(self):
        """
        Checks if the graph contains only non-negative edge weights.
        Raises a ValueError if any edge weight is negative.
        """
        for node, edges in self.graph.items():
            for neighbor, weight in edges.items():
                if weight < 0:
                    raise ValueError(f"Graph contains a negative weight edge: {node} -> {neighbor} ({weight})")

    def initialize_distances(self, start):
        """
        Initializes distance and predecessor dictionaries for the algorithm.
        :param start: The starting node.
        :return: Initialized dictionaries for distances and predecessors.
        """
        distances = {node: float('infinity') for node in self.graph}
        distances[start] = 0
        predecessors = {node: None for node in self.graph}
        return distances, predecessors

    def calculate_shortest_paths(self, start):
        """
        Calculates shortest paths from a start node using Dijkstra's algorithm.
        :param start: The starting node.
        :return: Dictionaries for distances and predecessors.
        """
        distances, predecessors = self.initialize_distances(start)
        priority_queue = [(0, start)]

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            if current_distance > distances[current_node]:
                continue

            for neighbor, weight in self.graph[current_node].items():
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_node
                    heapq.heappush(priority_queue, (distance, neighbor))

        return distances, predecessors

    def get_shortest_path(self, predecessors, start, end):
        """
        Constructs the shortest path from start to end node.
        :param predecessors: Dictionary of predecessors from calculate_shortest_paths.
        :param start: The starting node.
        :param end: The destination node.
        :return: List representing the shortest path from start to end.
        """
        path = []
        current = end

        while current is not None:
            path.append(current)
            current = predecessors[current]
        
        path.reverse()

        if path[0] == start:
            return path
        return None

    def display_shortest_paths(self, distances, predecessors, start):
        """
        Displays shortest distances and paths from the start node.
        :param distances: Distance dictionary from calculate_shortest_paths.
        :param predecessors: Predecessor dictionary from calculate_shortest_paths.
        :param start: The starting node.
        """
        print(f"Shortest paths from {start}:\n")
        for node in self.graph:
            if node != start:
                path = self.get_shortest_path(predecessors, start, node)
                if path:
                    print(f"Path to {node}: {' -> '.join(path)} (Distance: {distances[node]})")
                else:
                    print(f"No path to {node} found.")

    def add_edge(self, node1, node2, weight):
        """
        Adds an edge to the graph.
        :param node1: Starting node of the edge.
        :param node2: Ending node of the edge.
        :param weight: Weight of the edge.
        """
        if node1 not in self.graph:
            self.graph[node1] = {}
        self.graph[node1][node2] = weight

    def remove_edge(self, node1, node2):
        """
        Removes an edge from the graph.
        :param node1: Starting node of the edge.
        :param node2: Ending node of the edge.
        """
        if node1 in self.graph and node2 in self.graph[node1]:
            del self.graph[node1][node2]

    def display_graph(self):
        """
        Displays the entire graph with nodes and edges.
        """
        print("Graph structure:")
        for node, edges in self.graph.items():
            for neighbor, weight in edges.items():
                print(f"{node} -> {neighbor} [Weight: {weight}]")

    def shortest_path_between(self, start, end):
        """
        Calculates and displays the shortest path between two nodes.
        :param start: Starting node.
        :param end: Destination node.
        """
        distances, predecessors = self.calculate_shortest_paths(start)
        path = self.get_shortest_path(predecessors, start, end)
        if path:
            print(f"Shortest path from {start} to {end}: {' -> '.join(path)} (Distance: {distances[end]})")
        else:
            print(f"No path from {start} to {end} found.")

    def benchmark(self, start):
        """
        Benchmarks the performance of Dijkstra's algorithm on the current graph.
        :param start: The starting node.
        """
        start_time = time.time()
        distances, predecessors = self.calculate_shortest_paths(start)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Dijkstra's algorithm completed in {duration:.5f} seconds.")

    def show_all_nodes(self):
        """
        Prints all nodes in the graph.
        """
        print("Nodes in the graph:")
        for node in self.graph:
            print(node)

    def show_all_edges(self):
        """
        Prints all edges and their weights in the graph.
        """
        print("Edges in the graph:")
        for node, edges in self.graph.items():
            for neighbor, weight in edges.items():
                print(f"{node} -> {neighbor} [Weight: {weight}]")

# Example graph structure
graph = {
    'A': {'B': 4, 'C': 2},
    'B': {'A': 4, 'C': 1, 'D': 5},
    'C': {'A': 2, 'B': 1, 'D': 8, 'E': 10},
    'D': {'B': 5, 'C': 8, 'E': 2, 'Z': 6},
    'E': {'C': 10, 'D': 2, 'Z': 3},
    'Z': {}
}

# Initialize algorithm
dijkstra = DijkstraAlgorithm(graph)

# Display the graph structure
dijkstra.display_graph()

# Calculate shortest paths from a specific starting node
start_node = 'A'
distances, predecessors = dijkstra.calculate_shortest_paths(start_node)

# Display shortest paths from the start node to each node
dijkstra.display_shortest_paths(distances, predecessors, start_node)

# Adding a new edge and updating paths
dijkstra.add_edge('A', 'Z', 12)
print("\nAfter adding edge A -> Z with weight 12:")
distances, predecessors = dijkstra.calculate_shortest_paths(start_node)
dijkstra.display_shortest_paths(distances, predecessors, start_node)

# Updating an edge and finding shortest path between two nodes
dijkstra.add_edge('A', 'Z', 3)
print("\nAfter updating edge A -> Z with weight 3:")
distances, predecessors = dijkstra.calculate_shortest_paths(start_node)
dijkstra.display_shortest_paths(distances, predecessors, start_node)

# Benchmark performance
print("\nBenchmarking the algorithm's performance from node 'A':")
dijkstra.benchmark('A')

# Showing all nodes and edges in the graph
print("\nListing all nodes in the graph:")
dijkstra.show_all_nodes()
print("\nListing all edges in the graph:")
dijkstra.show_all_edges()
