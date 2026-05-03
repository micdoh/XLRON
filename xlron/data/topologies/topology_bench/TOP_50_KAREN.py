def create_karen_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {"lat": -41.29, "long": 174.78, "location": "Wellington", "country": "New Zealand"},
        2: {"lat": -45.87, "long": 170.5, "location": "Dunedin", "country": "New Zealand"},
        3: {"lat": -41.2, "long": 174.92, "location": "Lower Hutt", "country": "New Zealand"},
        4: {
            "lat": -40.35,
            "long": 175.62,
            "location": "Palmerston North",
            "country": "New Zealand",
        },
        5: {"lat": -43.65, "long": 172.48, "location": "Lincoln", "country": "New Zealand"},
        6: {"lat": -43.53, "long": 172.63, "location": "Christchurch", "country": "New Zealand"},
        7: {"lat": -45.85, "long": 170.38, "location": "Dunedin", "country": "New Zealand"},
        8: {"lat": -46.4, "long": 168.35, "location": "Invercargill", "country": "New Zealand"},
        9: {"lat": -44.02, "long": 170.5, "location": "Pleasant Point", "country": "New Zealand"},
        10: {"lat": -41.28, "long": 173.28, "location": "Nelson", "country": "New Zealand"},
        11: {"lat": -36.88, "long": 174.72, "location": "Auckland", "country": "New Zealand"},
        12: {"lat": -39.07, "long": 174.08, "location": "New Plymouth", "country": "New Zealand"},
        13: {"lat": -39.93, "long": 175.05, "location": "Wanganui", "country": "New Zealand"},
        14: {
            "lat": -40.35,
            "long": 175.62,
            "location": "Palmerston North",
            "country": "New Zealand",
        },
        15: {"lat": -36.8, "long": 174.75, "location": "North Shore", "country": "New Zealand"},
        16: {"lat": -36.4, "long": 174.67, "location": "Warkworth", "country": "New Zealand"},
        17: {"lat": -37.69, "long": 176.17, "location": "Tauranga", "country": "New Zealand"},
        18: {"lat": -38.65, "long": 178.0, "location": "Gisborne", "country": "New Zealand"},
        19: {"lat": -41.13, "long": 174.85, "location": "Porirua", "country": "New Zealand"},
        20: {"lat": -39.48, "long": 176.92, "location": "Napier", "country": "New Zealand"},
        21: {"lat": -38.14, "long": 176.25, "location": "Rotorua", "country": "New Zealand"},
        22: {"lat": -37.78, "long": 175.28, "location": "Hamilton", "country": "New Zealand"},
        23: {"lat": -36.87, "long": 174.77, "location": "Auckland", "country": "New Zealand"},
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 2.0, "fiber_length": 922.171188348746},
        2.0: {"source": 1.0, "destination": 3.0, "fiber_length": 23.09989849940613},
        3.0: {"source": 1.0, "destination": 4.0, "fiber_length": 189.2679888641153},
        4.0: {"source": 1.0, "destination": 6.0, "fiber_length": 457.8826929460702},
        5.0: {"source": 1.0, "destination": 10.0, "fiber_length": 188.0059942163219},
        6.0: {"source": 2.0, "destination": 5.0, "fiber_length": 438.2530618834782},
        7.0: {"source": 2.0, "destination": 7.0, "fiber_length": 14.332396675711},
        8.0: {"source": 2.0, "destination": 8.0, "fiber_length": 263.743125529436},
        9.0: {"source": 3.0, "destination": 19.0, "fiber_length": 14.61409143409326},
        10.0: {"source": 3.0, "destination": 14.0, "fiber_length": 167.0826641539101},
        11.0: {"source": 4.0, "destination": 23.0, "fiber_length": 590.9057529899435},
        12.0: {"source": 4.0, "destination": 12.0, "fiber_length": 290.8941997161656},
        13.0: {"source": 4.0, "destination": 13.0, "fiber_length": 100.9436691406827},
        14.0: {"source": 4.0, "destination": 14.0, "fiber_length": 0.0},
        15.0: {"source": 5.0, "destination": 6.0, "fiber_length": 26.99949097772463},
        16.0: {"source": 6.0, "destination": 9.0, "fiber_length": 269.2204917198632},
        17.0: {"source": 6.0, "destination": 10.0, "fiber_length": 383.7223635695941},
        18.0: {"source": 11.0, "destination": 23.0, "fiber_length": 6.876593997214164},
        19.0: {"source": 14.0, "destination": 20.0, "fiber_length": 220.7101604370363},
        20.0: {"source": 15.0, "destination": 16.0, "fiber_length": 67.57147702261867},
        21.0: {"source": 15.0, "destination": 23.0, "fiber_length": 11.97684780100587},
        22.0: {"source": 17.0, "destination": 21.0, "fiber_length": 75.79118533057107},
        23.0: {"source": 18.0, "destination": 20.0, "fiber_length": 196.7877799117688},
        24.0: {"source": 20.0, "destination": 21.0, "fiber_length": 239.8642203317239},
        25.0: {"source": 21.0, "destination": 22.0, "fiber_length": 140.9853245169645},
        26.0: {"source": 22.0, "destination": 23.0, "fiber_length": 166.1712876946465},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
