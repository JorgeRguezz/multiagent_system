import networkx as nx
import json

# ================================
# EDIT THESE PATHS
# ================================

video_kg_path_1 = "/home/gatv-projects/Desktop/project/knowledge_build_cache_Toda_la_historia_de_Ahri_League_of_Legends/graph_chunk_entity_relation.graphml"
video_kg_path_2 = "/home/gatv-projects/Desktop/project/playground/merged_kg.graphml"

output_graphml_path = "/home/gatv-projects/Desktop/project/playground/merged_kg.graphml"
output_json_path = "/home/gatv-projects/Desktop/project/playground/merged_kg.json"

# ================================
# Attribute merging logic
# ================================

def combine_values(v1, v2):
    """Combine two attribute values."""
    if v1 == v2:
        return v1

    list1 = v1 if isinstance(v1, list) else [v1]
    list2 = v2 if isinstance(v2, list) else [v2]

    merged = []

    for item in list1 + list2:
        if item not in merged:
            merged.append(item)

    return merged


def merge_attribute_dicts(d1, d2):
    """Merge two attribute dictionaries."""
    merged = {}

    keys = set(d1.keys()).union(d2.keys())

    for k in keys:
        if k in d1 and k in d2:
            merged[k] = combine_values(d1[k], d2[k])
        elif k in d1:
            merged[k] = d1[k]
        else:
            merged[k] = d2[k]

    return merged


# ================================
# KG merge function
# ================================

def merge_kgs_keep_all_attributes(G1, G2):

    merged = nx.compose(G1, G2)

    # Merge node attributes
    common_nodes = set(G1.nodes()).intersection(G2.nodes())

    for node in common_nodes:
        attrs = merge_attribute_dicts(G1.nodes[node], G2.nodes[node])
        merged.nodes[node].clear()
        merged.nodes[node].update(attrs)

    # Merge edge attributes
    common_edges = set(G1.edges()).intersection(G2.edges())

    for u, v in common_edges:
        attrs = merge_attribute_dicts(G1.edges[u, v], G2.edges[u, v])
        merged.edges[u, v].clear()
        merged.edges[u, v].update(attrs)

    return merged


# ================================
# GraphML sanitization
# ================================

def sanitize_graphml_attributes(G):
    """
    GraphML cannot store lists or dicts.
    Convert them to JSON strings.
    """

    def sanitize(value):

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        return json.dumps(value, ensure_ascii=False)

    # nodes
    for n, attrs in G.nodes(data=True):
        for k in list(attrs.keys()):
            attrs[k] = sanitize(attrs[k])

    # edges
    for u, v, attrs in G.edges(data=True):
        for k in list(attrs.keys()):
            attrs[k] = sanitize(attrs[k])

    return G


# ================================
# Main
# ================================

if __name__ == "__main__":

    print("Loading knowledge graphs...")

    kg1 = nx.read_graphml(video_kg_path_1)
    kg2 = nx.read_graphml(video_kg_path_2)

    print(f"KG1: {kg1.number_of_nodes()} nodes | {kg1.number_of_edges()} edges")
    print(f"KG2: {kg2.number_of_nodes()} nodes | {kg2.number_of_edges()} edges")

    print("\nMerging graphs...")

    merged_kg = merge_kgs_keep_all_attributes(kg1, kg2)

    print(f"Merged KG: {merged_kg.number_of_nodes()} nodes | {merged_kg.number_of_edges()} edges")

    # ================================
    # Save JSON (lossless)
    # ================================

    from networkx.readwrite import json_graph

    data = json_graph.node_link_data(merged_kg)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved lossless JSON KG → {output_json_path}")

    # ================================
    # Save GraphML
    # ================================

    graphml_copy = merged_kg.copy()
    sanitize_graphml_attributes(graphml_copy)

    nx.write_graphml(graphml_copy, output_graphml_path)

    print(f"Saved GraphML KG → {output_graphml_path}")

    print("\nDone.")