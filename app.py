from flask import Flask, render_template, jsonify
import networkx as nx
import random

app = Flask(__name__)

# Very Strong Perfect Graph: A type of graph where every induced subgraph 
# has an optimal coloring that equals its clique number.
# For simplicity, we assume a general graph for demonstration purposes.

# Create a graph with initial energy levels and edges.
def create_graph(num_nodes=20):
    G = nx.Graph()
    nodes = [(i, {"energy": 100.0}) for i in range(1, num_nodes + 1)]
    G.add_nodes_from(nodes)
    
    for i in range(1, num_nodes + 1):
        for j in range(i + 1, num_nodes + 1):
            if random.random() < 0.3:  # 30% probability of edge creation for better connectivity
                G.add_edge(i, j)
    
    return G

# Find all maximal cliques in the graph.
def get_maximal_cliques(graph):
    return list(nx.find_cliques(graph))

# Find strong independent sets from the maximal cliques.
def find_strong_independent_sets(graph, maximal_cliques):
    independent_sets = []
    for clique in maximal_cliques:
        independent_set = set(clique)
        independent_sets.append(independent_set)
    return independent_sets

# Select cluster heads from the strong independent sets.
# Cluster heads are nodes with the highest energy, and only the top 3 are selected.
def select_cluster_heads(graph, strong_independent_sets):
    potential_heads = set()
    for independent_set in strong_independent_sets:
        valid_nodes = [node for node in independent_set if graph.nodes[node]['energy'] > 0]
        if valid_nodes:
            max_energy_node = max(valid_nodes, key=lambda node: graph.nodes[node]['energy'])
            potential_heads.add(max_energy_node)
    # Sort potential heads by energy and take the top 3
    sorted_heads = sorted(potential_heads, key=lambda node: graph.nodes[node]['energy'], reverse=True)
    return sorted_heads[:3]

# Form clusters with the selected cluster heads.
# Each node joins the cluster of the nearest cluster head.
def form_clusters(graph, cluster_heads):
    clusters = {head: set() for head in cluster_heads}
    for node in graph.nodes():
        if node not in cluster_heads and graph.nodes[node]['energy'] > 0:
            reachable_heads = [head for head in cluster_heads if nx.has_path(graph, node, head)]
            if reachable_heads:
                closest_head = min(reachable_heads, key=lambda head: nx.shortest_path_length(graph, source=node, target=head))
                clusters[closest_head].add(node)
        elif graph.nodes[node]['energy'] > 0:
            clusters[node].add(node)
    return clusters

# Update the energy levels of nodes after each clustering round.
def update_energy(graph, clusters):
    for head, members in clusters.items():
        if graph.nodes[head]['energy'] > 0:
            graph.nodes[head]['energy'] = max(0, graph.nodes[head]['energy'] - 0.03)
        for member in members:
            if member != head and graph.nodes[member]['energy'] > 0:
                graph.nodes[member]['energy'] = max(0, graph.nodes[member]['energy'] - 0.01)

# Global graph object
G = create_graph(num_nodes=20)
positions = nx.spring_layout(G)  # Fixed positions for nodes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/graph')
def get_graph():
    graph_data = {
        'nodes': [{'id': n, 'energy': G.nodes[n]['energy'], 'pos': positions[n].tolist()} for n in G.nodes()],
        'edges': [{'source': u, 'target': v} for u, v in G.edges()]
    }
    return jsonify(graph_data)

@app.route('/clustering_process')
def clustering_process():
    if any(G.nodes[node]['energy'] > 0 for node in G.nodes):
        maximal_cliques = get_maximal_cliques(G)
        strong_independent_sets = find_strong_independent_sets(G, maximal_cliques)
        cluster_heads = select_cluster_heads(G, strong_independent_sets)
        clusters = form_clusters(G, cluster_heads)
        update_energy(G, clusters)
    
    graph_data = {
        'nodes': [{'id': n, 'energy': G.nodes[n]['energy'], 'pos': positions[n].tolist()} for n in G.nodes()],
        'edges': [{'source': u, 'target': v} for u, v in G.edges()],
        'cluster_heads': cluster_heads,
        'clusters': {head: list(members) for head, members in clusters.items()}
    }
    return jsonify(graph_data)

if __name__ == '__main__':
    app.run(debug=True)
