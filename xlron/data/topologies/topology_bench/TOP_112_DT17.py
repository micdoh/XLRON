def create_dt17_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {"lat": 53.5942039, "long": 7.2067435, "location": "Norden", "country": "Germany"},
        2: {"lat": 53.550341, "long": 10.000654, "location": "Altstadt", "country": "Germany"},
        3: {"lat": 53.0758196, "long": 8.8071646, "location": "Bremen", "country": "Germany"},
        4: {"lat": 52.5170365, "long": 13.3888599, "location": "Mitte", "country": "Germany"},
        5: {"lat": 52.3744779, "long": 9.7385532, "location": "Hannover", "country": "Germany"},
        6: {"lat": 51.4582235, "long": 7.0158171, "location": "Essen", "country": "Germany"},
        7: {"lat": 51.2254018, "long": 6.7763137, "location": "Dusseldorf", "country": "Germany"},
        8: {"lat": 43.7218277, "long": 0.9774958, "location": "Gimont", "country": "France"},
        9: {"lat": 51.5142273, "long": 7.4652789, "location": "Dortmund", "country": "Germany"},
        10: {"lat": 51.3406321, "long": 12.3747329, "location": "Leipzig", "country": "Germany"},
        11: {
            "lat": 50.1106444,
            "long": 8.6820917,
            "location": "Frankfurt am Main",
            "country": "Germany",
        },
        12: {"lat": 49.4892913, "long": 8.4673098, "location": "Mannheim", "country": "Germany"},
        13: {"lat": 49.0068705, "long": 8.4034195, "location": "Karlsruhe", "country": "Germany"},
        14: {"lat": 48.7784485, "long": 9.1800132, "location": "Stuttgart", "country": "Germany"},
        15: {"lat": 48.3966578, "long": 9.9931893, "location": "Ulm", "country": "Germany"},
        16: {"lat": 48.1371079, "long": 11.5753822, "location": "Munich", "country": "Germany"},
        17: {"lat": 49.453872, "long": 11.077298, "location": "Nuernberg", "country": "Germany"},
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 2.0, "fiber_length": 276.7953790040666},
        2.0: {"source": 2.0, "destination": 3.0, "fiber_length": 142.8552535606431},
        3.0: {"source": 3.0, "destination": 4.0, "fiber_length": 471.2857490145237},
        4.0: {"source": 2.0, "destination": 5.0, "fiber_length": 197.8844315072676},
        5.0: {"source": 5.0, "destination": 4.0, "fiber_length": 371.8185249786151},
        6.0: {"source": 5.0, "destination": 6.0, "fiber_length": 319.0608972317904},
        7.0: {"source": 6.0, "destination": 7.0, "fiber_length": 46.15941019417737},
        8.0: {"source": 7.0, "destination": 8.0, "fiber_length": 1410.942752920969},
        9.0: {"source": 8.0, "destination": 9.0, "fiber_length": 1489.032583718375},
        10.0: {"source": 9.0, "destination": 10.0, "fiber_length": 511.2872420793022},
        11.0: {"source": 10.0, "destination": 4.0, "fiber_length": 222.2100334388853},
        12.0: {"source": 9.0, "destination": 11.0, "fiber_length": 266.9204187111405},
        13.0: {"source": 11.0, "destination": 12.0, "fiber_length": 106.185033518311},
        14.0: {"source": 12.0, "destination": 13.0, "fiber_length": 80.76424751178835},
        15.0: {"source": 13.0, "destination": 14.0, "fiber_length": 93.29543906789593},
        16.0: {"source": 14.0, "destination": 15.0, "fiber_length": 110.0182349426603},
        17.0: {"source": 15.0, "destination": 16.0, "fiber_length": 180.9185771309096},
        18.0: {"source": 16.0, "destination": 17.0, "fiber_length": 226.3405130387763},
        19.0: {"source": 11.0, "destination": 17.0, "fiber_length": 280.2356298823898},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
