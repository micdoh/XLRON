def create_grena_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {"lat": 42.15, "long": 42.34, "location": "Samtredia", "country": "Georgia"},
        2: {"lat": 41.64, "long": 41.64, "location": "Batumi", "country": "Georgia"},
        3: {"lat": 42.25, "long": 42.7, "location": "Kutaisi", "country": "Georgia"},
        4: {"lat": 42.25, "long": 42.7, "location": "Kutaisi", "country": "Georgia"},
        5: {"lat": 42.15, "long": 41.67, "location": "P'ot'i", "country": "Georgia"},
        6: {"lat": 42.51, "long": 41.87, "location": "Zugdidi", "country": "Georgia"},
        7: {"lat": 41.99, "long": 43.6, "location": "Khashuri", "country": "Georgia"},
        8: {"lat": 41.69, "long": 44.83, "location": "Tbilisi", "country": "Georgia"},
        9: {"lat": 41.55, "long": 44.99, "location": "Rust'avi", "country": "Georgia"},
        10: {"lat": 41.92, "long": 45.47, "location": "Telavi", "country": "Georgia"},
        11: {"lat": 41.55, "long": 44.99, "location": "Rust'avi", "country": "Georgia"},
        12: {"lat": 41.98, "long": 44.12, "location": "Gori", "country": "Georgia"},
        13: {"lat": 42.25, "long": 42.7, "location": "Kutaisi", "country": "Georgia"},
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 5.0, "fiber_length": 82.85084409187695},
        2.0: {"source": 1.0, "destination": 13.0, "fiber_length": 47.50603436074783},
        3.0: {"source": 2.0, "destination": 5.0, "fiber_length": 85.14562276270968},
        4.0: {"source": 3.0, "destination": 4.0, "fiber_length": 0.0},
        5.0: {"source": 4.0, "destination": 13.0, "fiber_length": 0.0},
        6.0: {"source": 5.0, "destination": 6.0, "fiber_length": 64.91225821899334},
        7.0: {"source": 7.0, "destination": 12.0, "fiber_length": 64.49113307534793},
        8.0: {"source": 7.0, "destination": 13.0, "fiber_length": 119.4913221623684},
        9.0: {"source": 8.0, "destination": 10.0, "fiber_length": 88.33562198300541},
        10.0: {"source": 8.0, "destination": 11.0, "fiber_length": 30.71275645143643},
        11.0: {"source": 8.0, "destination": 12.0, "fiber_length": 100.6210366542427},
        12.0: {"source": 9.0, "destination": 11.0, "fiber_length": 0.0},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
