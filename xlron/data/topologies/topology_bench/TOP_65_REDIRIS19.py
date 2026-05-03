def create_rediris19_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {"lat": 42.82, "long": -1.64, "location": "Pamplona", "country": "Spain"},
        2: {"lat": 42.47, "long": -2.45, "location": "Logrono", "country": "Spain"},
        3: {"lat": 43.46, "long": -3.8, "location": "Santander", "country": "Spain"},
        4: {"lat": 43.0, "long": -2.75, "location": "Murguia", "country": "Spain"},
        5: {"lat": 39.57, "long": 2.65, "location": "Palma", "country": "Spain"},
        6: {"lat": 39.47, "long": -0.38, "location": "Valencia", "country": "Spain"},
        7: {"lat": 41.66, "long": -0.88, "location": "Zaragoza", "country": "Spain"},
        8: {"lat": 41.39, "long": 2.16, "location": "Barcelona", "country": "Spain"},
        9: {"lat": 37.98, "long": -1.12, "location": "Murcia", "country": "Spain"},
        10: {"lat": 42.88, "long": -8.55, "location": "Santiago de Compostela", "country": "Spain"},
        11: {"lat": 43.36, "long": -5.84, "location": "Oviedo", "country": "Spain"},
        12: {"lat": 38.88, "long": -6.97, "location": "Badajoz", "country": "Spain"},
        13: {"lat": 37.38, "long": -5.99, "location": "Sevilla", "country": "Spain"},
        14: {
            "lat": 28.1,
            "long": -15.42,
            "location": "Las Palmas de Gran Canaria",
            "country": "Spain",
        },
        15: {"lat": 28.48, "long": -16.32, "location": "La Laguna", "country": "Spain"},
        16: {"lat": 38.98, "long": -3.93, "location": "Ciudad Real", "country": "Spain"},
        17: {"lat": 40.42, "long": -3.7, "location": "City Center", "country": "Spain"},
        18: {"lat": 40.42, "long": -3.7, "location": "City Center", "country": "Spain"},
        19: {"lat": 41.65, "long": -4.72, "location": "Valladolid", "country": "Spain"},
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 4.0, "fiber_length": 138.8833257676686},
        2.0: {"source": 1.0, "destination": 7.0, "fiber_length": 215.0354958974909},
        3.0: {"source": 2.0, "destination": 19.0, "fiber_length": 312.5949971024991},
        4.0: {"source": 2.0, "destination": 7.0, "fiber_length": 236.7323288758541},
        5.0: {"source": 3.0, "destination": 11.0, "fiber_length": 247.736762447395},
        6.0: {"source": 3.0, "destination": 4.0, "fiber_length": 148.8913320614549},
        7.0: {"source": 4.0, "destination": 10.0, "fiber_length": 708.3410276977546},
        8.0: {"source": 4.0, "destination": 17.0, "fiber_length": 446.276915408953},
        9.0: {"source": 5.0, "destination": 6.0, "fiber_length": 390.1902263207722},
        10.0: {"source": 5.0, "destination": 8.0, "fiber_length": 309.8603628020256},
        11.0: {"source": 6.0, "destination": 9.0, "fiber_length": 266.5203351236332},
        12.0: {"source": 6.0, "destination": 13.0, "fiber_length": 811.509176028601},
        13.0: {"source": 6.0, "destination": 17.0, "fiber_length": 453.1083418251088},
        14.0: {"source": 6.0, "destination": 8.0, "fiber_length": 454.4362877620521},
        15.0: {"source": 7.0, "destination": 17.0, "fiber_length": 410.6179988648846},
        16.0: {"source": 7.0, "destination": 8.0, "fiber_length": 382.2518075886712},
        17.0: {"source": 8.0, "destination": 17.0, "fiber_length": 756.0643238408899},
        18.0: {"source": 9.0, "destination": 13.0, "fiber_length": 650.5307562517423},
        19.0: {"source": 10.0, "destination": 17.0, "fiber_length": 730.3673439066212},
        20.0: {"source": 10.0, "destination": 11.0, "fiber_length": 339.4882465912331},
        21.0: {"source": 10.0, "destination": 19.0, "fiber_length": 515.2798896622894},
        22.0: {"source": 12.0, "destination": 17.0, "fiber_length": 492.2167080047306},
        23.0: {"source": 12.0, "destination": 13.0, "fiber_length": 281.2889816134645},
        24.0: {"source": 13.0, "destination": 14.0, "fiber_length": 1694.827908318442},
        25.0: {"source": 13.0, "destination": 16.0, "fiber_length": 379.6705437058872},
        26.0: {"source": 13.0, "destination": 17.0, "fiber_length": 587.7008981762351},
        27.0: {"source": 14.0, "destination": 15.0, "fiber_length": 146.5929107863854},
        28.0: {"source": 15.0, "destination": 17.0, "fiber_length": 2196.691724693937},
        29.0: {"source": 16.0, "destination": 17.0, "fiber_length": 241.9875612160167},
        30.0: {"source": 17.0, "destination": 18.0, "fiber_length": 0.0},
        31.0: {"source": 17.0, "destination": 19.0, "fiber_length": 241.9804824971073},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
