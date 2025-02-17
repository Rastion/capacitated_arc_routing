import sys
import os
import random
from qubots.base_problem import BaseProblem

class CarpInstance:
    """
    Reads a CARP instance file following the DIMACS challenge format.
    """
    def read_elem(self, filename):
        with open(filename) as f:
            return [line.strip() for line in f.read().splitlines() if line.strip() != ""]
    
    def __init__(self, filename):
        file_it = iter(self.read_elem(filename))
        # Skip first two lines (header information)
        for _ in range(2):
            next(file_it)
        nb_nodes = int(next(file_it).split(":")[1].strip())
        self.nb_required_edges = int(next(file_it).split(":")[1].strip())
        nb_not_required_edges = int(next(file_it).split(":")[1].strip())
        self.nb_trucks = int(next(file_it).split(":")[1].strip())
        self.truck_capacity = int(next(file_it).split(":")[1].strip())
        # Skip next three lines
        for _ in range(3):
            next(file_it)
        
        self.demands_data = []
        self.costs_data = []
        self.origins_data = []
        self.destinations_data = []
        required_nodes = []
        nb_nodes_int = nb_nodes
        node_neighbors = [[0] * nb_nodes_int for _ in range(nb_nodes_int)]
        
        # Read required edges
        for _ in range(self.nb_required_edges):
            elements = next(file_it)
            # Expected format: "( x, y)   coste VALUE   demanda VALUE"
            parts = elements.split("   ")
            edge_str = parts[0].strip()  # e.g., "( 1, 2)"
            edge = tuple(map(int, edge_str.strip("()").split(",")))
            cost = int(parts[1].strip().split()[1])
            demand = int(parts[2].strip().split()[1])
            # For both directions
            self.costs_data.extend([cost, cost])
            self.demands_data.extend([demand, demand])
            self.origins_data.extend([edge[0], edge[1]])
            self.destinations_data.extend([edge[1], edge[0]])
            if edge[0] not in required_nodes:
                required_nodes.append(edge[0])
            if edge[1] not in required_nodes:
                required_nodes.append(edge[1])
            node_neighbors[edge[0] - 1][edge[1] - 1] = cost
            node_neighbors[edge[1] - 1][edge[0] - 1] = cost
        
        # Read non-required edges if any
        if nb_not_required_edges > 0:
            next(file_it)
            for _ in range(nb_not_required_edges):
                elements = next(file_it)
                parts = elements.split("   ")
                edge_str = parts[0].strip()
                edge = tuple(map(int, edge_str.strip("()").split(",")))
                cost = int(parts[1].strip().split()[1])
                node_neighbors[edge[0] - 1][edge[1] - 1] = cost
                node_neighbors[edge[1] - 1][edge[0] - 1] = cost
        
        depot_line = next(file_it)
        self.depot = int(depot_line.split(":")[1].strip())
        
        # Compute distances between nodes using Dijkstra's algorithm
        self.required_nodes = required_nodes
        self.edges_dist_data = [[None] * (2 * self.nb_required_edges) for _ in range(2 * self.nb_required_edges)]
        for i in range(2 * self.nb_required_edges):
            for j in range(2 * self.nb_required_edges):
                if self.destinations_data[i] == self.origins_data[j]:
                    self.edges_dist_data[i][j] = 0
                else:
                    dists = self._dijkstra(self.destinations_data[i] - 1, nb_nodes_int, node_neighbors)
                    self.edges_dist_data[i][j] = dists[self.origins_data[j] - 1]
        
        self.dist_to_depot_data = [None] * (2 * self.nb_required_edges)
        for i in range(2 * self.nb_required_edges):
            if self.destinations_data[i] == self.depot:
                self.dist_to_depot_data[i] = 0
            else:
                dists = self._dijkstra(self.destinations_data[i] - 1, nb_nodes_int, node_neighbors)
                self.dist_to_depot_data[i] = dists[self.depot - 1]
        
        self.dist_from_depot_data = [None] * (2 * self.nb_required_edges)
        for i in range(2 * self.nb_required_edges):
            if self.origins_data[i] == self.depot:
                self.dist_from_depot_data[i] = 0
            else:
                dists = self._dijkstra(self.depot - 1, nb_nodes_int, node_neighbors)
                self.dist_from_depot_data[i] = dists[self.origins_data[i] - 1]
    
    def _dijkstra(self, start, nb_nodes, neighbors):
        import heapq
        dist = [sys.maxsize] * nb_nodes
        dist[start] = 0
        visited = [False] * nb_nodes
        heap = [(0, start)]
        while heap:
            d, u = heapq.heappop(heap)
            if visited[u]:
                continue
            visited[u] = True
            for v in range(nb_nodes):
                if neighbors[u][v] > 0 and not visited[v]:
                    if dist[u] + neighbors[u][v] < dist[v]:
                        dist[v] = dist[u] + neighbors[u][v]
                        heapq.heappush(heap, (dist[v], v))
        return dist

class CapacitatedArcRoutingProblem(BaseProblem):
    """
    Capacitated Arc Routing Problem (CARP):

    A fleet of vehicles with uniform capacity must service a set of required edges
    (each with a specified demand and cost). Each required edge (which can be traversed
    in either direction) must be serviced by exactly one vehicle. Vehicles start and end
    at a common depot. The objective is to minimize the total distance traveled by all vehicles,
    while ensuring that the total demand serviced by each vehicle does not exceed its capacity.

    Candidate solution representation:
      A dictionary with key "edge_sequences" mapping to a list (one per vehicle) of lists.
      Each inner list represents the sequence (order) of edge indices (0-indexed, in the range
      [0, 2*nb_required_edges - 1]) serviced by that vehicle.
    """
    def __init__(self, instance_file=None, carp_instance=None):
        if instance_file is not None:
            self.instance = CarpInstance(instance_file)
        elif carp_instance is not None:
            self.instance = carp_instance
        else:
            raise ValueError("Either 'instance_file' or 'carp_instance' must be provided.")
    
    def evaluate_solution(self, solution) -> float:
        PENALTY = 1e9
        inst = self.instance
        # Expect solution to be a dict with key "edge_sequences" (a list of routes, one per truck)
        if not isinstance(solution, dict) or "edge_sequences" not in solution:
            return PENALTY
        edge_sequences = solution["edge_sequences"]
        if len(edge_sequences) != inst.nb_trucks:
            return PENALTY
        
        # Check that every required edge is serviced exactly once.
        served = [False] * (2 * inst.nb_required_edges)
        for route in edge_sequences:
            for e in route:
                if e < 0 or e >= 2 * inst.nb_required_edges:
                    return PENALTY
                if served[e]:
                    return PENALTY
                served[e] = True
        # For each required edge i, exactly one of {2*i, 2*i+1} must be served.
        for i in range(inst.nb_required_edges):
            if not ((served[2*i] and not served[2*i+1]) or (served[2*i+1] and not served[2*i])):
                return PENALTY
        
        total_distance = 0
        # Evaluate each vehicle's route.
        for route in edge_sequences:
            c = len(route)
            # Capacity: sum of demands must not exceed truck capacity.
            route_demand = sum(inst.demands_data[e] for e in route)
            if route_demand > inst.truck_capacity:
                return PENALTY
            route_distance = 0
            if c > 0:
                # Distance from depot to first edge's origin plus cost of first edge.
                route_distance += inst.costs_data[route[0]] + inst.dist_from_depot_data[route[0]]
                # For consecutive edges, add cost and the inter-edge distance.
                for i in range(1, c):
                    route_distance += inst.costs_data[route[i]] + inst.edges_dist_data[route[i-1]][route[i]]
                # Add distance from last edge's destination back to depot.
                route_distance += inst.dist_to_depot_data[route[c-1]]
            total_distance += route_distance
        return total_distance
    
    def random_solution(self):
        inst = self.instance
        # For each required edge, randomly select one of its two directions.
        selected_edges = []
        for i in range(inst.nb_required_edges):
            if random.random() < 0.5:
                selected_edges.append(2*i)
            else:
                selected_edges.append(2*i+1)
        # Shuffle the selected edges.
        random.shuffle(selected_edges)
        # Randomly assign edges to trucks.
        edge_sequences = [[] for _ in range(inst.nb_trucks)]
        for e in selected_edges:
            truck = random.randint(0, inst.nb_trucks - 1)
            edge_sequences[truck].append(e)
        return {"edge_sequences": edge_sequences}