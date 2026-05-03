def create_gambia_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {"lat": 13.49, "long": -16.09, "location": "Kerewan", "country": "Gambia (the)"},
        2: {"lat": 13.45, "long": -16.58, "location": "Banjul", "country": "Gambia (the)"},
        3: {"lat": 13.32, "long": -14.22, "location": "Basse Santa Su", "country": "Gambia (the)"},
        4: {"lat": 13.43, "long": -14.65, "location": "Bansang", "country": "Gambia (the)"},
        5: {"lat": 13.4, "long": -16.66, "location": "Abuko", "country": "Gambia (the)"},
        6: {"lat": 13.44, "long": -16.68, "location": "Bakau", "country": "Gambia (the)"},
        7: {"lat": 13.46, "long": -16.71, "location": "Bakau", "country": "Gambia (the)"},
        8: {"lat": 13.48, "long": -16.68, "location": "Bakau", "country": "Gambia (the)"},
        9: {"lat": 13.27, "long": -16.65, "location": "Brikama", "country": "Gambia (the)"},
        10: {"lat": 13.36, "long": -16.69, "location": "Sukuta", "country": "Gambia (the)"},
        11: {"lat": 13.43, "long": -15.53, "location": "Soma", "country": "Gambia (the)"},
        12: {"lat": 13.57, "long": -15.6, "location": "Farafenni", "country": "Gambia (the)"},
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 2.0, "fiber_length": 79.7596016272376},
        2.0: {"source": 1.0, "destination": 12.0, "fiber_length": 80.57265131939667},
        3.0: {"source": 2.0, "destination": 6.0, "fiber_length": 16.30764112863728},
        4.0: {"source": 3.0, "destination": 11.0, "fiber_length": 213.3617100826682},
        5.0: {"source": 3.0, "destination": 4.0, "fiber_length": 72.14726175977378},
        6.0: {"source": 5.0, "destination": 6.0, "fiber_length": 7.418895033817087},
        7.0: {"source": 6.0, "destination": 7.0, "fiber_length": 5.900088072205524},
        8.0: {"source": 6.0, "destination": 8.0, "fiber_length": 6.671695598673639},
        9.0: {"source": 6.0, "destination": 9.0, "fiber_length": 28.76962336783309},
        10.0: {"source": 6.0, "destination": 10.0, "fiber_length": 13.44167574214923},
        11.0: {"source": 10.0, "destination": 11.0, "fiber_length": 188.5773787521576},
        12.0: {"source": 11.0, "destination": 12.0, "fiber_length": 25.96447188829978},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
