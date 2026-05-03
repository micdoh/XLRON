def create_rnp_brazil_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {"lat": -3.7275, "long": -38.5275, "location": "Fortaleza", "country": "Brazil"},
        2: {"lat": -8.0539, "long": -34.8808, "location": "Recife", "country": "Brazil"},
        3: {"lat": -12.9831, "long": -38.4928, "location": "Salvador, Bahia", "country": "Brazil"},
        4: {"lat": -19.9281, "long": -43.9419, "location": "Belo Horizonte", "country": "Brazil"},
        5: {"lat": -22.9111, "long": -43.2056, "location": "Rio de Janeiro", "country": "Brazil"},
        6: {"lat": -23.5504, "long": -46.6339, "location": "São Paulo", "country": "Brazil"},
        7: {"lat": -15.7939, "long": -47.8828, "location": "Brasília", "country": "Brazil"},
        8: {"lat": -25.4297, "long": -49.2719, "location": "Curitiba", "country": "Brazil"},
        9: {"lat": -27.5933, "long": -48.553, "location": "Florianópolis", "country": "Brazil"},
        10: {"lat": -30.0328, "long": -51.23, "location": "Porto Alegre", "country": "Brazil"},
    }

    edge_attributes = {
        1.0: {"source": 10.0, "destination": 7.0, "fiber_length": 2024.62070937385},
        2.0: {"source": 10.0, "destination": 9.0, "fiber_length": 564.417052013471},
        3.0: {"source": 9.0, "destination": 8.0, "fiber_length": 376.48304920594},
        4.0: {"source": 6.0, "destination": 8.0, "fiber_length": 508.483952770296},
        5.0: {"source": 6.0, "destination": 7.0, "fiber_length": 1308.4809813383},
        6.0: {"source": 6.0, "destination": 5.0, "fiber_length": 536.14854787129},
        7.0: {"source": 7.0, "destination": 4.0, "fiber_length": 930.947827468676},
        8.0: {"source": 4.0, "destination": 5.0, "fiber_length": 510.504423926675},
        9.0: {"source": 5.0, "destination": 3.0, "fiber_length": 1513.69036343304},
        10.0: {"source": 3.0, "destination": 2.0, "fiber_length": 1013.18861023336},
        11.0: {"source": 2.0, "destination": 1.0, "fiber_length": 941.596919009031},
        12.0: {"source": 1.0, "destination": 4.0, "fiber_length": 2368.32822145231},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
