def create_atmnet_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {
            "lat": 40.76,
            "long": -111.89,
            "location": "Salt Lake City",
            "country": "United States of America (the)",
        },
        2: {
            "lat": 44.98,
            "long": -93.26,
            "location": "Minneapolis",
            "country": "United States of America (the)",
        },
        3: {
            "lat": 39.11,
            "long": -94.63,
            "location": "Kansas City",
            "country": "United States of America (the)",
        },
        4: {
            "lat": 39.74,
            "long": -104.98,
            "location": "Denver",
            "country": "United States of America (the)",
        },
        5: {
            "lat": 40.44,
            "long": -80.0,
            "location": "Pittsburgh",
            "country": "United States of America (the)",
        },
        6: {
            "lat": 39.95,
            "long": -75.16,
            "location": "Philadelphia",
            "country": "United States of America (the)",
        },
        7: {
            "lat": 41.85,
            "long": -87.65,
            "location": "Chicago",
            "country": "United States of America (the)",
        },
        8: {
            "lat": 42.33,
            "long": -83.05,
            "location": "Detroit",
            "country": "United States of America (the)",
        },
        9: {
            "lat": 40.71,
            "long": -74.01,
            "location": "New York City",
            "country": "United States of America (the)",
        },
        10: {
            "lat": 38.9,
            "long": -77.04,
            "location": "Washington, D.C.",
            "country": "United States of America (the)",
        },
        11: {
            "lat": 34.05,
            "long": -118.24,
            "location": "Los Angeles",
            "country": "United States of America (the)",
        },
        12: {
            "lat": 29.76,
            "long": -95.36,
            "location": "Houston",
            "country": "United States of America (the)",
        },
        13: {
            "lat": 38.63,
            "long": -90.2,
            "location": "St. Louis",
            "country": "United States of America (the)",
        },
        14: {
            "lat": 47.61,
            "long": -122.33,
            "location": "Seattle",
            "country": "United States of America (the)",
        },
        15: {
            "lat": 37.8,
            "long": -122.27,
            "location": "Oakland",
            "country": "United States of America (the)",
        },
        16: {
            "lat": 37.35,
            "long": -121.96,
            "location": "Santa Clara",
            "country": "United States of America (the)",
        },
        17: {
            "lat": 33.75,
            "long": -84.39,
            "location": "Atlanta",
            "country": "United States of America (the)",
        },
        18: {
            "lat": 32.72,
            "long": -117.16,
            "location": "San Diego",
            "country": "United States of America (the)",
        },
        19: {
            "lat": 33.45,
            "long": -112.07,
            "location": "Phoenix",
            "country": "United States of America (the)",
        },
        20: {
            "lat": 32.22,
            "long": -110.93,
            "location": "Tucson",
            "country": "United States of America (the)",
        },
        21: {
            "lat": 32.78,
            "long": -96.81,
            "location": "Dallas",
            "country": "United States of America (the)",
        },
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 4.0, "fiber_length": 895.6985411380621},
        2.0: {"source": 1.0, "destination": 15.0, "fiber_length": 1427.072737779968},
        3.0: {"source": 2.0, "destination": 7.0, "fiber_length": 856.6967858075002},
        4.0: {"source": 3.0, "destination": 4.0, "fiber_length": 1336.875843009606},
        5.0: {"source": 3.0, "destination": 13.0, "fiber_length": 580.7625411994072},
        6.0: {"source": 5.0, "destination": 6.0, "fiber_length": 621.9499312218177},
        7.0: {"source": 5.0, "destination": 8.0, "fiber_length": 494.9717198278396},
        8.0: {"source": 6.0, "destination": 9.0, "fiber_length": 193.5165266446945},
        9.0: {"source": 7.0, "destination": 13.0, "fiber_length": 627.5025611615271},
        10.0: {"source": 7.0, "destination": 8.0, "fiber_length": 574.8944829350417},
        11.0: {"source": 9.0, "destination": 10.0, "fiber_length": 491.7553096743694},
        12.0: {"source": 10.0, "destination": 17.0, "fiber_length": 1308.123444881084},
        13.0: {"source": 11.0, "destination": 18.0, "fiber_length": 268.0139459776118},
        14.0: {"source": 11.0, "destination": 16.0, "fiber_length": 746.0789546600618},
        15.0: {"source": 12.0, "destination": 17.0, "fiber_length": 1500.0},
        16.0: {"source": 12.0, "destination": 13.0, "fiber_length": 1500.0},
        17.0: {"source": 12.0, "destination": 21.0, "fiber_length": 544.4600499700148},
        18.0: {"source": 14.0, "destination": 15.0, "fiber_length": 1500.0},
        19.0: {"source": 15.0, "destination": 16.0, "fiber_length": 85.51486469975411},
        20.0: {"source": 18.0, "destination": 19.0, "fiber_length": 721.5882097187069},
        21.0: {"source": 19.0, "destination": 20.0, "fiber_length": 260.0209332979543},
        22.0: {"source": 20.0, "destination": 21.0, "fiber_length": 1655.833630654037},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
