def create_nobel_germany_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {"lat": 52.39, "long": 9.8, "location": "Hannover", "country": "Germany"},
        2: {"lat": 50.14, "long": 8.66, "location": "Frankfurt am Main", "country": "Germany"},
        3: {"lat": 53.55, "long": 10.08, "location": "Marienthal", "country": "Germany"},
        4: {"lat": 53.6, "long": 7.21, "location": "Norden", "country": "Germany"},
        5: {"lat": 53.08, "long": 8.8, "location": "Bremen", "country": "Germany"},
        6: {"lat": 52.52, "long": 13.48, "location": "Fennpfuhl", "country": "Germany"},
        7: {"lat": 48.15, "long": 11.55, "location": "Munich", "country": "Germany"},
        8: {"lat": 48.4, "long": 9.99, "location": "Ulm", "country": "Germany"},
        9: {"lat": 49.45, "long": 11.08, "location": "Nuernberg", "country": "Germany"},
        10: {
            "lat": 48.73,
            "long": 9.12,
            "location": "Leinfelden-Echterdingen",
            "country": "Germany",
        },
        11: {"lat": 49.01, "long": 8.41, "location": "Karlsruhe", "country": "Germany"},
        12: {"lat": 49.49, "long": 8.49, "location": "Mannheim", "country": "Germany"},
        13: {"lat": 51.44, "long": 7.0, "location": "Essen", "country": "Germany"},
        14: {"lat": 51.51, "long": 7.48, "location": "Dortmund", "country": "Germany"},
        15: {"lat": 51.22, "long": 6.78, "location": "Dusseldorf", "country": "Germany"},
        16: {"lat": 50.92, "long": 7.01, "location": "Humboldtkolonie", "country": "Germany"},
        17: {"lat": 51.34, "long": 12.38, "location": "Leipzig", "country": "Germany"},
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 6.0, "fiber_length": 374.624782328087},
        2.0: {"source": 1.0, "destination": 5.0, "fiber_length": 153.1132171736606},
        3.0: {"source": 1.0, "destination": 14.0, "fiber_length": 280.0268896201081},
        4.0: {"source": 1.0, "destination": 2.0, "fiber_length": 393.6775619461023},
        5.0: {"source": 1.0, "destination": 3.0, "fiber_length": 195.5122974449201},
        6.0: {"source": 1.0, "destination": 17.0, "fiber_length": 318.2231036629754},
        7.0: {"source": 2.0, "destination": 16.0, "fiber_length": 218.0056692643854},
        8.0: {"source": 2.0, "destination": 17.0, "fiber_length": 440.6546566372959},
        9.0: {"source": 2.0, "destination": 12.0, "fiber_length": 109.9479513076914},
        10.0: {"source": 2.0, "destination": 9.0, "fiber_length": 284.8248893317208},
        11.0: {"source": 3.0, "destination": 6.0, "fiber_length": 381.7847325471307},
        12.0: {"source": 3.0, "destination": 5.0, "fiber_length": 149.7064069006946},
        13.0: {"source": 4.0, "destination": 5.0, "fiber_length": 180.53361104788},
        14.0: {"source": 4.0, "destination": 14.0, "fiber_length": 349.6690242570837},
        15.0: {"source": 6.0, "destination": 17.0, "fiber_length": 227.0072937943405},
        16.0: {"source": 7.0, "destination": 9.0, "fiber_length": 222.8924142009059},
        17.0: {"source": 7.0, "destination": 8.0, "fiber_length": 178.1210043987085},
        18.0: {"source": 8.0, "destination": 10.0, "fiber_length": 110.683858131905},
        19.0: {"source": 9.0, "destination": 17.0, "fiber_length": 344.1972217356784},
        20.0: {"source": 9.0, "destination": 10.0, "fiber_length": 245.4570856667369},
        21.0: {"source": 10.0, "destination": 11.0, "fiber_length": 90.82158654545206},
        22.0: {"source": 11.0, "destination": 12.0, "fiber_length": 80.53273587887958},
        23.0: {"source": 13.0, "destination": 14.0, "fiber_length": 51.21455844269055},
        24.0: {"source": 13.0, "destination": 15.0, "fiber_length": 43.26842092186249},
        25.0: {"source": 14.0, "destination": 16.0, "fiber_length": 109.9782254493657},
        26.0: {"source": 15.0, "destination": 16.0, "fiber_length": 55.54142781966004},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
