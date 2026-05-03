def create_memorex_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {"lat": 50.1106444, "long": 8.6820917, "location": "Frankfurt", "country": nan},
        2: {"lat": 49.7913804, "long": 10.0246112, "location": "Wurzburg", "country": nan},
        3: {"lat": 49.453872, "long": 11.077298, "location": "Nuremburg", "country": nan},
        4: {"lat": 48.7784485, "long": 9.1800132, "location": "Stuttgart", "country": nan},
        5: {"lat": 48.1371079, "long": 11.5753822, "location": "Munich", "country": nan},
        6: {"lat": 50.0596288, "long": 14.446459273258, "location": "Prague", "country": nan},
        7: {"lat": 49.1922443, "long": 16.6113382, "location": "Brno", "country": nan},
        8: {"lat": 48.15926025, "long": 17.1396586914216, "location": "Bratislava", "country": nan},
        9: {"lat": 48.2083537, "long": 16.3725042, "location": "Vienna", "country": nan},
        10: {"lat": 48.3059078, "long": 14.286198, "location": "Linz", "country": nan},
        11: {"lat": 47.7981346, "long": 13.0464806, "location": "Salzburg", "country": nan},
        12: {"lat": 47.8701474, "long": 12.6423403, "location": "Transtein", "country": nan},
        13: {"lat": 47.4978789, "long": 19.0402383, "location": "Budapest", "country": nan},
        14: {"lat": 47.2654296, "long": 11.3927685, "location": "Innsbruck", "country": nan},
        15: {"lat": 46.6167284, "long": 13.8500268, "location": "Villach", "country": nan},
        16: {"lat": 46.623943, "long": 14.3075976, "location": "klagenfurt", "country": nan},
        17: {"lat": 47.0708678, "long": 15.4382786, "location": "Graz", "country": nan},
        18: {"lat": 46.0500268, "long": 14.5069289, "location": "Ljubljana", "country": nan},
        19: {"lat": 45.4384958, "long": 10.9924122, "location": "Verona", "country": nan},
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 2.0, "fiber_length": 153.6},
        2.0: {"source": 1.0, "destination": 4.0, "fiber_length": 228.66},
        3.0: {"source": 2.0, "destination": 3.0, "fiber_length": 126.915},
        4.0: {"source": 3.0, "destination": 5.0, "fiber_length": 226.335},
        5.0: {"source": 5.0, "destination": 4.0, "fiber_length": 285.72},
        6.0: {"source": 5.0, "destination": 6.0, "fiber_length": 448.425},
        7.0: {"source": 5.0, "destination": 12.0, "fiber_length": 127.125},
        8.0: {"source": 6.0, "destination": 7.0, "fiber_length": 275.01},
        9.0: {"source": 7.0, "destination": 8.0, "fiber_length": 181.86},
        10.0: {"source": 8.0, "destination": 13.0, "fiber_length": 239.7},
        11.0: {"source": 8.0, "destination": 9.0, "fiber_length": 85.71000000000001},
        12.0: {"source": 9.0, "destination": 13.0, "fiber_length": 321.21},
        13.0: {"source": 9.0, "destination": 17.0, "fiber_length": 216.84},
        14.0: {"source": 9.0, "destination": 10.0, "fiber_length": 232.245},
        15.0: {"source": 10.0, "destination": 11.0, "fiber_length": 162.105},
        16.0: {"source": 11.0, "destination": 15.0, "fiber_length": 217.065},
        17.0: {"source": 11.0, "destination": 12.0, "fiber_length": 46.815},
        18.0: {"source": 12.0, "destination": 14.0, "fiber_length": 173.055},
        19.0: {"source": 14.0, "destination": 19.0, "fiber_length": 308.19},
        20.0: {"source": 15.0, "destination": 16.0, "fiber_length": 52.425},
        21.0: {"source": 19.0, "destination": 15.0, "fiber_length": 384.84},
        22.0: {"source": 16.0, "destination": 18.0, "fiber_length": 98.445},
        23.0: {"source": 16.0, "destination": 17.0, "fiber_length": 148.98},
        24.0: {"source": 17.0, "destination": 18.0, "fiber_length": 201.0},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
