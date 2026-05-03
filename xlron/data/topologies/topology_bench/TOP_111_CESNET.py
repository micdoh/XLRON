def create_cesnet_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {"lat": -1.3716486, "long": 133.8214949, "location": "Warmare", "country": "Indonesia"},
        2: {"lat": 50.7702648, "long": 15.0583947, "location": "Liberec", "country": "Czechia"},
        3: {
            "lat": 49.7072278,
            "long": 15.2954608,
            "location": "Ledec nad Sazavou",
            "country": "Czechia",
        },
        4: {"lat": 50.0874654, "long": 14.4212535, "location": "Prague", "country": "Czechia"},
        5: {"lat": 50.0385812, "long": 15.7791356, "location": "Pardubice", "country": "Czechia"},
        6: {"lat": 49.7477415, "long": 13.3775249, "location": "Pilsen", "country": "Czechia"},
        7: {
            "lat": 48.9747357,
            "long": 14.474285,
            "location": "Ceske Budejovice",
            "country": "Czechia",
        },
        8: {"lat": 49.396064, "long": 15.5903065, "location": "Jihlava", "country": "Czechia"},
        9: {"lat": 49.1922443, "long": 16.6113382, "location": "Brno", "country": "Czechia"},
        10: {"lat": 49.5940567, "long": 17.251143, "location": "Olomouc", "country": "Czechia"},
        11: {"lat": 49.226766, "long": 17.6667415, "location": "Zlin", "country": "Czechia"},
        12: {"lat": 49.8349139, "long": 18.2820084, "location": "Ostrava", "country": "Czechia"},
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 2.0, "fiber_length": 15126.7894450579},
        2.0: {"source": 2.0, "destination": 3.0, "fiber_length": 179.1007254590727},
        3.0: {"source": 3.0, "destination": 4.0, "fiber_length": 113.3310665405402},
        4.0: {"source": 4.0, "destination": 5.0, "fiber_length": 145.6168629246583},
        5.0: {"source": 4.0, "destination": 6.0, "fiber_length": 125.5983775804574},
        6.0: {"source": 6.0, "destination": 7.0, "fiber_length": 175.5463841547675},
        7.0: {"source": 7.0, "destination": 8.0, "fiber_length": 140.501016942214},
        8.0: {"source": 8.0, "destination": 9.0, "fiber_length": 116.1509529157739},
        9.0: {"source": 9.0, "destination": 10.0, "fiber_length": 96.51784566585519},
        10.0: {"source": 10.0, "destination": 11.0, "fiber_length": 76.07251199436654},
        11.0: {"source": 11.0, "destination": 12.0, "fiber_length": 121.3466350545247},
        12.0: {"source": 9.0, "destination": 12.0, "fiber_length": 210.2850452866539},
        13.0: {"source": 4.0, "destination": 8.0, "fiber_length": 170.8088147370161},
        14.0: {"source": 5.0, "destination": 3.0, "fiber_length": 75.8792625138322},
        15.0: {"source": 10.0, "destination": 3.0, "fiber_length": 212.0282438358674},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
