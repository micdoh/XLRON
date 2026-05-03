def create_marwan_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {"lat": 31.63, "long": -8.01, "location": "Marrakesh", "country": "Morocco"},
        2: {"lat": 34.04, "long": -5.0, "location": "Fes", "country": "Morocco"},
        3: {"lat": 34.01, "long": -6.83, "location": "Rabat", "country": "Morocco"},
        4: {"lat": 30.42, "long": -9.6, "location": "Agadir", "country": "Morocco"},
        5: {"lat": 35.78, "long": -5.81, "location": "Tangier", "country": "Morocco"},
        6: {"lat": 33.59, "long": -7.62, "location": "Casablanca", "country": "Morocco"},
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 2.0, "fiber_length": 582.627043544287},
        2.0: {"source": 1.0, "destination": 4.0, "fiber_length": 303.9301435674128},
        3.0: {"source": 2.0, "destination": 5.0, "fiber_length": 310.6430120065281},
        4.0: {"source": 3.0, "destination": 4.0, "fiber_length": 714.9962255186151},
        5.0: {"source": 3.0, "destination": 6.0, "fiber_length": 129.9864336338506},
        6.0: {"source": 5.0, "destination": 6.0, "fiber_length": 441.6242356589813},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
