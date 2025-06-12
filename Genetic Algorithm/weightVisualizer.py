# weight_visualizer.py
import networkx as nx
import torch
from matplotlib import pyplot as plt


def visualize_model(model):
    G = nx.DiGraph()

    layers = []
    for layer in model:
        if isinstance(layer, torch.nn.Linear):
            layers.append(layer)

    # Düğümleri oluştur
    node_id = 0
    node_layers = []
    for layer_idx, layer in enumerate(layers):
        nodes = []
        for i in range(layer.in_features):
            G.add_node(f"L{layer_idx}_in_{i}", layer=layer_idx)
            nodes.append(f"L{layer_idx}_in_{i}")
        for i in range(layer.out_features):
            G.add_node(f"L{layer_idx}_out_{i}", layer=layer_idx + 1)
            nodes.append(f"L{layer_idx}_out_{i}")
        node_layers.append(nodes)

    # Kenarları ve ağırlıkları ekle
    for idx, layer in enumerate(layers):
        weights = layer.weight.data.numpy()
        in_nodes = [n for n in G.nodes if n.startswith(f"L{idx}_in")]
        out_nodes = [n for n in G.nodes if n.startswith(f"L{idx}_out")]

        for i, out_node in enumerate(out_nodes):
            for j, in_node in enumerate(in_nodes):
                weight = weights[i][j]
                # Kenar kalınlığını ağırlık mutlak değeriyle ayarla
                G.add_edge(in_node, out_node, weight=abs(weight))

    pos = {}
    # Pozisyonları ayarla (katmanlara göre yatay dizilim)
    layer_counts = {}
    for node in G.nodes:
        layer_idx = G.nodes[node]['layer']
        layer_counts.setdefault(layer_idx, 0)
        pos[node] = (layer_idx * 3, layer_counts[layer_idx])
        layer_counts[layer_idx] += 1

    edges = G.edges()
    weights = [G[u][v]['weight']*5 for u,v in edges]  # Kalınlık için ölçekle

    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', width=weights)
    plt.title("Model ağırlık bağlantıları (kalınlık ağırlık ile orantılı)")
    plt.show()

