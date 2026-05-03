def create_japan12_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {"lat": 43.075, "long": 141.34, "location": "Sapporo", "country": "Japan"},
        2: {"lat": 38.27, "long": 140.85, "location": "Sendai", "country": "Japan"},
        3: {"lat": 35.68, "long": 139.77, "location": "Tokyo", "country": "Japan"},
        4: {"lat": 35.66, "long": 139.28, "location": "Hachioji", "country": "Japan"},
        5: {"lat": 36.58, "long": 136.65, "location": "Kanazawa", "country": "Japan"},
        6: {"lat": 36.66, "long": 138.19, "location": "Nagano", "country": "Japan"},
        7: {"lat": 35.149, "long": 136.91, "location": "Nagoya", "country": "Japan"},
        8: {"lat": 34.67, "long": 135.5, "location": "Osaka", "country": "Japan"},
        9: {"lat": 34.39, "long": 132.449, "location": "Hiroshima", "country": "Japan"},
        10: {"lat": 33.84, "long": 132.75, "location": "Matsuyama", "country": "Japan"},
        11: {"lat": 33.57, "long": 130.35, "location": "Hakata", "country": "Japan"},
        12: {"lat": 26.21, "long": 127.68, "location": "Naha", "country": "Japan"},
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 2.0, "fiber_length": 889.9499999999999},
        2.0: {"source": 1.0, "destination": 4.0, "fiber_length": 1570.5},
        3.0: {"source": 2.0, "destination": 3.0, "fiber_length": 527.7},
        4.0: {"source": 3.0, "destination": 4.0, "fiber_length": 71.1},
        5.0: {"source": 3.0, "destination": 7.0, "fiber_length": 549.0},
        6.0: {"source": 4.0, "destination": 6.0, "fiber_length": 376.05},
        7.0: {"source": 6.0, "destination": 5.0, "fiber_length": 378.3},
        8.0: {"source": 6.0, "destination": 7.0, "fiber_length": 376.2},
        9.0: {"source": 7.0, "destination": 8.0, "fiber_length": 279.9},
        10.0: {"source": 5.0, "destination": 8.0, "fiber_length": 395.7},
        11.0: {"source": 7.0, "destination": 10.0, "fiber_length": 736.05},
        12.0: {"source": 8.0, "destination": 9.0, "fiber_length": 512.4000000000001},
        13.0: {"source": 9.0, "destination": 10.0, "fiber_length": 99.30000000000001},
        14.0: {"source": 9.0, "destination": 11.0, "fiber_length": 421.05},
        15.0: {"source": 10.0, "destination": 12.0, "fiber_length": 1500.0},
        16.0: {"source": 10.0, "destination": 11.0, "fiber_length": 547.5},
        17.0: {"source": 11.0, "destination": 12.0, "fiber_length": 1367.85},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
