def create_japan25_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {"lat": 43.075, "long": 141.34, "location": "Sapporo", "country": "Japan"},
        2: {"lat": 38.27, "long": 140.85, "location": "Sendai", "country": "Japan"},
        3: {"lat": 37.77, "long": 140.46, "location": "Fukushima", "country": "Japan"},
        4: {"lat": 36.37, "long": 140.47, "location": "Mito", "country": "Japan"},
        5: {"lat": 36.57, "long": 139.88, "location": "Utsunomiya", "country": "Japan"},
        6: {"lat": 36.4, "long": 139.06, "location": "Maebashi", "country": "Japan"},
        7: {"lat": 35.9, "long": 139.63, "location": "Omiya", "country": "Japan"},
        8: {"lat": 35.6, "long": 140.16, "location": "Chiba", "country": "Japan"},
        9: {"lat": 35.68, "long": 139.77, "location": "Tokyo", "country": "Japan"},
        10: {"lat": 35.66, "long": 139.28, "location": "Hachioji", "country": "Japan"},
        11: {"lat": 35.458, "long": 139.64, "location": "Yokohama", "country": "Japan"},
        12: {"lat": 37.92, "long": 139.05, "location": "Niigata", "country": "Japan"},
        13: {"lat": 36.58, "long": 136.65, "location": "Kanazawa", "country": "Japan"},
        14: {"lat": 36.66, "long": 138.19, "location": "Nagano", "country": "Japan"},
        15: {"lat": 35.45, "long": 136.76, "location": "Gifu", "country": "Japan"},
        16: {"lat": 35.013, "long": 138.4, "location": "Shizuoka", "country": "Japan"},
        17: {"lat": 35.149, "long": 136.91, "location": "Nagoya", "country": "Japan"},
        18: {"lat": 35.018, "long": 135.76, "location": "Kyoto", "country": "Japan"},
        19: {"lat": 34.67, "long": 135.5, "location": "Osaka", "country": "Japan"},
        20: {"lat": 34.69, "long": 135.18, "location": "Kobe", "country": "Japan"},
        21: {"lat": 34.39, "long": 132.449, "location": "Hiroshima", "country": "Japan"},
        22: {"lat": 33.84, "long": 132.75, "location": "Matsuyama", "country": "Japan"},
        23: {"lat": 33.57, "long": 130.35, "location": "Hakata", "country": "Japan"},
        24: {"lat": 32.81, "long": 130.7, "location": "Kumamoto", "country": "Japan"},
        25: {"lat": 26.21, "long": 127.68, "location": "Naha", "country": "Japan"},
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 2.0, "fiber_length": 889.9499999999999},
        2.0: {"source": 1.0, "destination": 12.0, "fiber_length": 1396.65},
        3.0: {"source": 2.0, "destination": 3.0, "fiber_length": 118.5},
        4.0: {"source": 2.0, "destination": 4.0, "fiber_length": 368.1},
        5.0: {"source": 3.0, "destination": 5.0, "fiber_length": 244.95},
        6.0: {"source": 3.0, "destination": 12.0, "fiber_length": 270.15},
        7.0: {"source": 4.0, "destination": 5.0, "fiber_length": 143.4},
        8.0: {"source": 4.0, "destination": 7.0, "fiber_length": 175.5},
        9.0: {"source": 4.0, "destination": 8.0, "fiber_length": 191.25},
        10.0: {"source": 5.0, "destination": 6.0, "fiber_length": 159.75},
        11.0: {"source": 5.0, "destination": 7.0, "fiber_length": 118.8},
        12.0: {"source": 6.0, "destination": 7.0, "fiber_length": 112.05},
        13.0: {"source": 6.0, "destination": 10.0, "fiber_length": 144.6},
        14.0: {"source": 6.0, "destination": 12.0, "fiber_length": 343.35},
        15.0: {"source": 6.0, "destination": 14.0, "fiber_length": 176.1},
        16.0: {"source": 7.0, "destination": 8.0, "fiber_length": 99.14999999999999},
        17.0: {"source": 7.0, "destination": 9.0, "fiber_length": 45.45},
        18.0: {"source": 8.0, "destination": 9.0, "fiber_length": 58.8},
        19.0: {"source": 9.0, "destination": 10.0, "fiber_length": 71.1},
        20.0: {"source": 9.0, "destination": 11.0, "fiber_length": 43.2},
        21.0: {"source": 10.0, "destination": 11.0, "fiber_length": 54.75},
        22.0: {"source": 10.0, "destination": 14.0, "fiber_length": 376.05},
        23.0: {"source": 11.0, "destination": 16.0, "fiber_length": 227.1},
        24.0: {"source": 12.0, "destination": 14.0, "fiber_length": 316.95},
        25.0: {"source": 13.0, "destination": 14.0, "fiber_length": 378.3},
        26.0: {"source": 13.0, "destination": 18.0, "fiber_length": 337.2},
        27.0: {"source": 14.0, "destination": 17.0, "fiber_length": 376.2},
        28.0: {"source": 15.0, "destination": 17.0, "fiber_length": 45.45},
        29.0: {"source": 15.0, "destination": 18.0, "fiber_length": 175.95},
        30.0: {"source": 16.0, "destination": 17.0, "fiber_length": 278.7},
        31.0: {"source": 17.0, "destination": 19.0, "fiber_length": 312.0},
        32.0: {"source": 18.0, "destination": 19.0, "fiber_length": 58.5},
        33.0: {"source": 18.0, "destination": 20.0, "fiber_length": 116.1},
        34.0: {"source": 19.0, "destination": 20.0, "fiber_length": 55.34999999999999},
        35.0: {"source": 19.0, "destination": 22.0, "fiber_length": 616.2},
        36.0: {"source": 20.0, "destination": 21.0, "fiber_length": 457.05},
        37.0: {"source": 21.0, "destination": 22.0, "fiber_length": 99.30000000000001},
        38.0: {"source": 21.0, "destination": 23.0, "fiber_length": 421.05},
        39.0: {"source": 22.0, "destination": 23.0, "fiber_length": 547.5},
        40.0: {"source": 22.0, "destination": 24.0, "fiber_length": 471.75},
        41.0: {"source": 23.0, "destination": 24.0, "fiber_length": 177.6},
        42.0: {"source": 23.0, "destination": 25.0, "fiber_length": 1367.85},
        43.0: {"source": 24.0, "destination": 25.0, "fiber_length": 1266.3},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
