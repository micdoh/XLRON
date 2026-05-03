def create_savvis_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {
            "lat": 40.71,
            "long": -74.01,
            "location": "New York City",
            "country": "United States of America (the)",
        },
        2: {
            "lat": 39.95,
            "long": -75.16,
            "location": "Philadelphia",
            "country": "United States of America (the)",
        },
        3: {
            "lat": 42.33,
            "long": -83.05,
            "location": "Detroit",
            "country": "United States of America (the)",
        },
        4: {
            "lat": 40.44,
            "long": -80.0,
            "location": "Pittsburgh",
            "country": "United States of America (the)",
        },
        5: {
            "lat": 29.76,
            "long": -95.36,
            "location": "Houston",
            "country": "United States of America (the)",
        },
        6: {
            "lat": 30.27,
            "long": -97.74,
            "location": "Austin",
            "country": "United States of America (the)",
        },
        7: {
            "lat": 38.9,
            "long": -77.04,
            "location": "Washington, D.C.",
            "country": "United States of America (the)",
        },
        8: {
            "lat": 33.75,
            "long": -84.39,
            "location": "Atlanta",
            "country": "United States of America (the)",
        },
        9: {
            "lat": 32.78,
            "long": -96.81,
            "location": "Dallas",
            "country": "United States of America (the)",
        },
        10: {
            "lat": 40.12,
            "long": -88.24,
            "location": "Champaign",
            "country": "United States of America (the)",
        },
        11: {
            "lat": 41.85,
            "long": -87.65,
            "location": "Chicago",
            "country": "United States of America (the)",
        },
        12: {
            "lat": 37.77,
            "long": -122.42,
            "location": "San Francisco",
            "country": "United States of America (the)",
        },
        13: {
            "lat": 34.05,
            "long": -118.24,
            "location": "Los Angeles",
            "country": "United States of America (the)",
        },
        14: {
            "lat": 32.72,
            "long": -117.16,
            "location": "San Diego",
            "country": "United States of America (the)",
        },
        15: {
            "lat": 33.45,
            "long": -112.07,
            "location": "Phoenix",
            "country": "United States of America (the)",
        },
        16: {
            "lat": 40.76,
            "long": -111.89,
            "location": "Salt Lake City",
            "country": "United States of America (the)",
        },
        17: {
            "lat": 39.74,
            "long": -104.98,
            "location": "Denver",
            "country": "United States of America (the)",
        },
        18: {
            "lat": 39.11,
            "long": -94.63,
            "location": "Kansas City",
            "country": "United States of America (the)",
        },
        19: {
            "lat": 38.63,
            "long": -90.2,
            "location": "St. Louis",
            "country": "United States of America (the)",
        },
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 2.0, "fiber_length": 193.5165266446945},
        2.0: {"source": 1.0, "destination": 4.0, "fiber_length": 760.0480661195877},
        3.0: {"source": 2.0, "destination": 7.0, "fiber_length": 298.8881926608235},
        4.0: {"source": 3.0, "destination": 11.0, "fiber_length": 574.8944829350417},
        5.0: {"source": 3.0, "destination": 4.0, "fiber_length": 494.9717198278396},
        6.0: {"source": 5.0, "destination": 6.0, "fiber_length": 354.0915074929874},
        7.0: {"source": 5.0, "destination": 8.0, "fiber_length": 1500.0},
        8.0: {"source": 6.0, "destination": 9.0, "fiber_length": 439.0260901564217},
        9.0: {"source": 7.0, "destination": 8.0, "fiber_length": 1308.123444881084},
        10.0: {"source": 9.0, "destination": 19.0, "fiber_length": 1323.434150934681},
        11.0: {"source": 9.0, "destination": 15.0, "fiber_length": 1777.377955221904},
        12.0: {"source": 10.0, "destination": 19.0, "fiber_length": 354.4151322501527},
        13.0: {"source": 11.0, "destination": 19.0, "fiber_length": 627.5025611615271},
        14.0: {"source": 12.0, "destination": 13.0, "fiber_length": 838.7553033988663},
        15.0: {"source": 12.0, "destination": 16.0, "fiber_length": 1447.204027572812},
        16.0: {"source": 13.0, "destination": 14.0, "fiber_length": 268.0139459776118},
        17.0: {"source": 13.0, "destination": 15.0, "fiber_length": 861.3692628497527},
        18.0: {"source": 16.0, "destination": 17.0, "fiber_length": 895.6985411380621},
        19.0: {"source": 17.0, "destination": 18.0, "fiber_length": 1336.875843009606},
        20.0: {"source": 18.0, "destination": 19.0, "fiber_length": 580.7625411994072},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
