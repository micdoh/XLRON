def create_via_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {"lat": 38.7077507, "long": -9.1365919, "location": "Lisbon", "country": "Portugal"},
        2: {"lat": 40.4167047, "long": -3.7035825, "location": "Madrid", "country": "Spain"},
        3: {"lat": 41.3828939, "long": 2.1774322, "location": "Barcelona", "country": "Spain"},
        4: {"lat": 48.8588897, "long": 2.32004102172007, "location": "Paris", "country": "France"},
        5: {
            "lat": 51.4893335,
            "long": -0.144055084527687,
            "location": "London",
            "country": "United Kingdom",
        },
        6: {
            "lat": 52.3730796,
            "long": 4.8924534,
            "location": "Amsterdam",
            "country": "Netherlands",
        },
        7: {"lat": 50.1106444, "long": 8.6820917, "location": "Frankfurt", "country": "Germany"},
        8: {"lat": 48.1371079, "long": 11.5753822, "location": "Munich", "country": "Germany"},
        9: {"lat": 52.5170365, "long": 13.3888599, "location": "Berlin", "country": "Germany"},
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 2.0, "fiber_length": 754.3499999999999},
        2.0: {"source": 1.0, "destination": 5.0, "fiber_length": 1980.1},
        3.0: {"source": 2.0, "destination": 3.0, "fiber_length": 758.58},
        4.0: {"source": 3.0, "destination": 4.0, "fiber_length": 1247.055},
        5.0: {"source": 4.0, "destination": 5.0, "fiber_length": 511.575},
        6.0: {"source": 4.0, "destination": 7.0, "fiber_length": 720.03},
        7.0: {"source": 5.0, "destination": 6.0, "fiber_length": 538.425},
        8.0: {"source": 5.0, "destination": 7.0, "fiber_length": 957.765},
        9.0: {"source": 6.0, "destination": 7.0, "fiber_length": 546.66},
        10.0: {"source": 7.0, "destination": 8.0, "fiber_length": 456.105},
        11.0: {"source": 7.0, "destination": 9.0, "fiber_length": 633.72},
        12.0: {"source": 8.0, "destination": 9.0, "fiber_length": 755.5649999999999},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
