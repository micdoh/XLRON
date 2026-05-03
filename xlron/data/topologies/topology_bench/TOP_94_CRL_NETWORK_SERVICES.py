def create_crl_network_services_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {
            "lat": 33.45,
            "long": -112.07,
            "location": "Phoenix",
            "country": "United States of America (the)",
        },
        2: {
            "lat": 35.08,
            "long": -106.65,
            "location": "Albuquerque",
            "country": "United States of America (the)",
        },
        3: {
            "lat": 33.84,
            "long": -117.91,
            "location": "Anaheim",
            "country": "United States of America (the)",
        },
        4: {
            "lat": 32.72,
            "long": -117.16,
            "location": "San Diego",
            "country": "United States of America (the)",
        },
        5: {
            "lat": 29.42,
            "long": -98.49,
            "location": "San Antonio",
            "country": "United States of America (the)",
        },
        6: {
            "lat": 29.76,
            "long": -95.36,
            "location": "Houston",
            "country": "United States of America (the)",
        },
        7: {
            "lat": 32.78,
            "long": -96.81,
            "location": "Dallas",
            "country": "United States of America (the)",
        },
        8: {
            "lat": 30.27,
            "long": -97.74,
            "location": "Austin",
            "country": "United States of America (the)",
        },
        9: {
            "lat": 33.75,
            "long": -84.39,
            "location": "Atlanta",
            "country": "United States of America (the)",
        },
        10: {
            "lat": 35.77,
            "long": -78.64,
            "location": "Raleigh",
            "country": "United States of America (the)",
        },
        11: {
            "lat": 39.11,
            "long": -94.63,
            "location": "Kansas City",
            "country": "United States of America (the)",
        },
        12: {
            "lat": 39.74,
            "long": -104.98,
            "location": "Denver",
            "country": "United States of America (the)",
        },
        13: {
            "lat": 27.95,
            "long": -82.46,
            "location": "Tampa",
            "country": "United States of America (the)",
        },
        14: {
            "lat": 38.63,
            "long": -90.2,
            "location": "St. Louis",
            "country": "United States of America (the)",
        },
        15: {
            "lat": 40.44,
            "long": -80.0,
            "location": "Pittsburgh",
            "country": "United States of America (the)",
        },
        16: {
            "lat": 40.71,
            "long": -74.01,
            "location": "New York City",
            "country": "United States of America (the)",
        },
        17: {
            "lat": 39.95,
            "long": -75.16,
            "location": "Philadelphia",
            "country": "United States of America (the)",
        },
        18: {
            "lat": 39.29,
            "long": -76.61,
            "location": "Baltimore",
            "country": "United States of America (the)",
        },
        19: {
            "lat": 38.9,
            "long": -77.04,
            "location": "Washington, D.C.",
            "country": "United States of America (the)",
        },
        20: {
            "lat": 41.85,
            "long": -87.65,
            "location": "Chicago",
            "country": "United States of America (the)",
        },
        21: {
            "lat": 42.33,
            "long": -83.05,
            "location": "Detroit",
            "country": "United States of America (the)",
        },
        22: {
            "lat": 41.5,
            "long": -81.7,
            "location": "Cleveland",
            "country": "United States of America (the)",
        },
        23: {
            "lat": 42.36,
            "long": -71.06,
            "location": "Boston",
            "country": "United States of America (the)",
        },
        24: {
            "lat": 37.34,
            "long": -121.89,
            "location": "San Jose",
            "country": "United States of America (the)",
        },
        25: {
            "lat": 34.05,
            "long": -118.24,
            "location": "Los Angeles",
            "country": "United States of America (the)",
        },
        26: {
            "lat": 47.61,
            "long": -122.33,
            "location": "Seattle",
            "country": "United States of America (the)",
        },
        27: {
            "lat": 45.52,
            "long": -122.68,
            "location": "Portland",
            "country": "United States of America (the)",
        },
        28: {
            "lat": 38.44,
            "long": -122.71,
            "location": "Santa Rosa",
            "country": "United States of America (the)",
        },
        29: {
            "lat": 38.58,
            "long": -121.49,
            "location": "Sacramento",
            "country": "United States of America (the)",
        },
        30: {
            "lat": 37.96,
            "long": -121.29,
            "location": "Stockton",
            "country": "United States of America (the)",
        },
        31: {
            "lat": 43.21,
            "long": -71.54,
            "location": "Concord",
            "country": "United States of America (the)",
        },
        32: {
            "lat": 37.97,
            "long": -122.53,
            "location": "San Rafael",
            "country": "United States of America (the)",
        },
        33: {
            "lat": 37.77,
            "long": -122.42,
            "location": "San Francisco",
            "country": "United States of America (the)",
        },
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 2.0, "fiber_length": 794.9053687405473},
        2.0: {"source": 1.0, "destination": 4.0, "fiber_length": 721.5882097187069},
        3.0: {"source": 2.0, "destination": 7.0, "fiber_length": 1414.060458882298},
        4.0: {"source": 3.0, "destination": 25.0, "fiber_length": 57.54795592014905},
        5.0: {"source": 3.0, "destination": 4.0, "fiber_length": 214.0861789294347},
        6.0: {"source": 5.0, "destination": 6.0, "fiber_length": 457.4870157403198},
        7.0: {"source": 5.0, "destination": 8.0, "fiber_length": 178.528161273931},
        8.0: {"source": 6.0, "destination": 7.0, "fiber_length": 544.4600499700148},
        9.0: {"source": 7.0, "destination": 9.0, "fiber_length": 1500.0},
        10.0: {"source": 7.0, "destination": 14.0, "fiber_length": 1323.434150934681},
        11.0: {"source": 7.0, "destination": 8.0, "fiber_length": 439.0260901564217},
        12.0: {"source": 9.0, "destination": 10.0, "fiber_length": 856.7329203000172},
        13.0: {"source": 9.0, "destination": 13.0, "fiber_length": 1006.029022461507},
        14.0: {"source": 10.0, "destination": 19.0, "fiber_length": 563.5055664441111},
        15.0: {"source": 11.0, "destination": 12.0, "fiber_length": 1336.875843009606},
        16.0: {"source": 11.0, "destination": 14.0, "fiber_length": 580.7625411994072},
        17.0: {"source": 12.0, "destination": 33.0, "fiber_length": 1906.958816527235},
        18.0: {"source": 14.0, "destination": 20.0, "fiber_length": 627.5025611615271},
        19.0: {"source": 15.0, "destination": 19.0, "fiber_length": 458.6437394496578},
        20.0: {"source": 15.0, "destination": 22.0, "fiber_length": 277.6478760498783},
        21.0: {"source": 16.0, "destination": 17.0, "fiber_length": 193.5165266446945},
        22.0: {"source": 16.0, "destination": 23.0, "fiber_length": 459.728758316418},
        23.0: {"source": 17.0, "destination": 18.0, "fiber_length": 216.3834396565364},
        24.0: {"source": 18.0, "destination": 19.0, "fiber_length": 85.61340860642284},
        25.0: {"source": 20.0, "destination": 26.0, "fiber_length": 3485.643998403856},
        26.0: {"source": 20.0, "destination": 21.0, "fiber_length": 574.8944829350417},
        27.0: {"source": 21.0, "destination": 22.0, "fiber_length": 217.3432784660523},
        28.0: {"source": 22.0, "destination": 23.0, "fiber_length": 1327.170812389139},
        29.0: {"source": 24.0, "destination": 25.0, "fiber_length": 738.4958532321912},
        30.0: {"source": 24.0, "destination": 33.0, "fiber_length": 100.2750862366293},
        31.0: {"source": 26.0, "destination": 27.0, "fiber_length": 350.8979570264834},
        32.0: {"source": 27.0, "destination": 28.0, "fiber_length": 1180.895953716089},
        33.0: {"source": 28.0, "destination": 29.0, "fiber_length": 160.9300921413741},
        34.0: {"source": 28.0, "destination": 32.0, "fiber_length": 81.86537296101017},
        35.0: {"source": 29.0, "destination": 30.0, "fiber_length": 106.676040609737},
        36.0: {"source": 30.0, "destination": 31.0, "fiber_length": 5224.086131262762},
        37.0: {"source": 31.0, "destination": 33.0, "fiber_length": 5347.898728593499},
        38.0: {"source": 32.0, "destination": 33.0, "fiber_length": 36.36694868522626},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
