import networkx as nx
import json

# ================================
# EDIT THESE PATHS
# ================================

video_kg_path_1 = "/home/gatv-projects/Desktop/project/playground/lol_knowledge_graph_full.graphml"
video_kg_path_2 = "/home/gatv-projects/Desktop/project/knowledge_sanitization/cache/sanitized_global/graph_AetherNexus.graphml"

output_graphml_path = "/home/gatv-projects/Desktop/project/playground/test_merged_kg.graphml"
output_json_path = "/home/gatv-projects/Desktop/project/playground/test_merged_kg.json"

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


def normalize_node_key(node):
    """Canonicalize node ids so equivalent entities from both KGs collide."""
    return " ".join(str(node).strip().upper().split())


def normalize_graph_node_keys(G):
    """Return a copy of G with normalized node ids."""
    mapping = {node: normalize_node_key(node) for node in G.nodes()}
    return nx.relabel_nodes(G, mapping, copy=True)


def project_node_attributes(node_id, attrs):
    """Reduce merged node attributes to the final schema expected by this test."""
    node_title = attrs.get("title")
    if not node_title:
        node_title = f"<b>{node_id}</b>"

    projected = {
        "entity_type": attrs.get("entity_type", attrs.get("type")),
        "title": node_title,
        "size": attrs.get("size"),
        "description": attrs.get("description"),
    }

    if "source_id" in attrs:
        projected["source_id"] = attrs["source_id"]

    return {k: v for k, v in projected.items() if v is not None}


# ================================
# KG merge function
# ================================

def _normalized_merge_graph_type(G1, G2):
    """Pick a graph class that can represent both inputs without losing edges."""
    if G1.is_directed() or G2.is_directed():
        return nx.MultiDiGraph
    return nx.MultiGraph


def _coerce_graph(G, target_type):
    """Copy a graph into the target NetworkX graph type."""
    if isinstance(G, target_type):
        return G.copy()
    return target_type(G)


def _edge_ids(G):
    """Return comparable edge identifiers for simple and multi graphs."""
    if G.is_multigraph():
        return set(G.edges(keys=True))
    return set(G.edges())


def merge_kgs_keep_all_attributes(G1, G2):
    G1 = normalize_graph_node_keys(G1)
    G2 = normalize_graph_node_keys(G2)

    graph_type = _normalized_merge_graph_type(G1, G2)
    G1 = _coerce_graph(G1, graph_type)
    G2 = _coerce_graph(G2, graph_type)

    merged = nx.compose(G1, G2)

    # Merge node attributes
    common_nodes = set(G1.nodes()).intersection(G2.nodes())

    for node in common_nodes:
        attrs = merge_attribute_dicts(G1.nodes[node], G2.nodes[node])
        merged.nodes[node].clear()
        merged.nodes[node].update(attrs)

    for node, attrs in merged.nodes(data=True):
        projected_attrs = project_node_attributes(node, dict(attrs))
        merged.nodes[node].clear()
        merged.nodes[node].update(projected_attrs)

    # Merge edge attributes
    common_edges = _edge_ids(G1).intersection(_edge_ids(G2))

    if merged.is_multigraph():
        for u, v, key in common_edges:
            attrs = merge_attribute_dicts(G1.edges[u, v, key], G2.edges[u, v, key])
            merged.edges[u, v, key].clear()
            merged.edges[u, v, key].update(attrs)
    else:
        for u, v in common_edges:
            attrs = merge_attribute_dicts(G1.edges[u, v], G2.edges[u, v])
            merged.edges[u, v].clear()
            merged.edges[u, v].update(attrs)

    stats = {
        "merged_node_count": len(common_nodes),
        "kg1_unique_node_count": G1.number_of_nodes() - len(common_nodes),
        "kg2_unique_node_count": G2.number_of_nodes() - len(common_nodes),
    }

    return merged, stats


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

    normalized_kg1_nodes = {normalize_node_key(node) for node in kg1.nodes()}
    normalized_kg2_nodes = {normalize_node_key(node) for node in kg2.nodes()}
    normalized_overlap = normalized_kg1_nodes.intersection(normalized_kg2_nodes)

    print(f"Normalized shared node keys before merge: {len(normalized_overlap)}")

    print("\nMerging graphs...")

    merged_kg, merge_stats = merge_kgs_keep_all_attributes(kg1, kg2)

    print(f"Merged KG: {merged_kg.number_of_nodes()} nodes | {merged_kg.number_of_edges()} edges")
    print(
        "Merged nodes shared by KG1 and KG2: "
        f"{merge_stats['merged_node_count']}"
    )
    print(f"KG1-only nodes: {merge_stats['kg1_unique_node_count']}")
    print(f"KG2-only nodes: {merge_stats['kg2_unique_node_count']}")

    # ================================
    # Save JSON (lossless)
    # ================================

    from networkx.readwrite import json_graph

    data = json_graph.node_link_data(merged_kg, edges="links")

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
