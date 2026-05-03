def create_digex_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {
            "lat": 41.85,
            "long": -87.65,
            "location": "Chicago",
            "country": "United States of America (the)",
        },
        2: {
            "lat": 41.26,
            "long": -95.94,
            "location": "Omaha",
            "country": "United States of America (the)",
        },
        3: {
            "lat": 39.96,
            "long": -83.0,
            "location": "Columbus",
            "country": "United States of America (the)",
        },
        4: {
            "lat": 42.33,
            "long": -83.05,
            "location": "Detroit",
            "country": "United States of America (the)",
        },
        5: {
            "lat": 33.75,
            "long": -84.39,
            "location": "Atlanta",
            "country": "United States of America (the)",
        },
        6: {
            "lat": 30.33,
            "long": -81.66,
            "location": "Jacksonville",
            "country": "United States of America (the)",
        },
        7: {
            "lat": 39.11,
            "long": -94.63,
            "location": "Kansas City",
            "country": "United States of America (the)",
        },
        8: {
            "lat": 38.63,
            "long": -90.2,
            "location": "St. Louis",
            "country": "United States of America (the)",
        },
        9: {
            "lat": 28.54,
            "long": -81.38,
            "location": "Orlando",
            "country": "United States of America (the)",
        },
        10: {
            "lat": 27.95,
            "long": -82.46,
            "location": "Tampa",
            "country": "United States of America (the)",
        },
        11: {
            "lat": 39.29,
            "long": -76.61,
            "location": "Baltimore",
            "country": "United States of America (the)",
        },
        12: {
            "lat": 39.95,
            "long": -75.16,
            "location": "Philadelphia",
            "country": "United States of America (the)",
        },
        13: {
            "lat": 40.22,
            "long": -74.74,
            "location": "Trenton",
            "country": "United States of America (the)",
        },
        14: {
            "lat": 40.44,
            "long": -80.0,
            "location": "Pittsburgh",
            "country": "United States of America (the)",
        },
        15: {
            "lat": 37.55,
            "long": -77.46,
            "location": "Richmond",
            "country": "United States of America (the)",
        },
        16: {
            "lat": 35.23,
            "long": -80.84,
            "location": "Charlotte",
            "country": "United States of America (the)",
        },
        17: {
            "lat": 25.77,
            "long": -80.19,
            "location": "Miami",
            "country": "United States of America (the)",
        },
        18: {
            "lat": 40.74,
            "long": -74.17,
            "location": "Newark",
            "country": "United States of America (the)",
        },
        19: {
            "lat": 40.71,
            "long": -74.01,
            "location": "New York City",
            "country": "United States of America (the)",
        },
        20: {
            "lat": 38.9,
            "long": -77.04,
            "location": "Washington, D.C.",
            "country": "United States of America (the)",
        },
        21: {
            "lat": 42.36,
            "long": -71.06,
            "location": "Boston",
            "country": "United States of America (the)",
        },
        22: {
            "lat": 38.25,
            "long": -85.76,
            "location": "Louisville",
            "country": "United States of America (the)",
        },
        23: {
            "lat": 39.16,
            "long": -84.46,
            "location": "Cincinnati",
            "country": "United States of America (the)",
        },
        24: {
            "lat": 37.67,
            "long": -122.08,
            "location": "Hayward",
            "country": "United States of America (the)",
        },
        25: {
            "lat": 34.05,
            "long": -118.24,
            "location": "Los Angeles",
            "country": "United States of America (the)",
        },
        26: {
            "lat": 32.78,
            "long": -96.81,
            "location": "Dallas",
            "country": "United States of America (the)",
        },
        27: {
            "lat": 32.73,
            "long": -97.32,
            "location": "Fort Worth",
            "country": "United States of America (the)",
        },
        28: {
            "lat": 30.27,
            "long": -97.74,
            "location": "Austin",
            "country": "United States of America (the)",
        },
        29: {
            "lat": 37.77,
            "long": -122.42,
            "location": "San Francisco",
            "country": "United States of America (the)",
        },
        30: {
            "lat": 29.76,
            "long": -95.36,
            "location": "Houston",
            "country": "United States of America (the)",
        },
        31: {
            "lat": 36.17,
            "long": -86.78,
            "location": "Nashville",
            "country": "United States of America (the)",
        },
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 2.0, "fiber_length": 1038.965784439191},
        2.0: {"source": 1.0, "destination": 4.0, "fiber_length": 574.8944829350417},
        3.0: {"source": 1.0, "destination": 8.0, "fiber_length": 627.5025611615271},
        4.0: {"source": 2.0, "destination": 24.0, "fiber_length": 2837.825839206324},
        5.0: {"source": 3.0, "destination": 4.0, "fiber_length": 395.3478240391674},
        6.0: {"source": 3.0, "destination": 14.0, "fiber_length": 390.4603456578371},
        7.0: {"source": 3.0, "destination": 23.0, "fiber_length": 230.3234448820241},
        8.0: {"source": 5.0, "destination": 30.0, "fiber_length": 1500.0},
        9.0: {"source": 5.0, "destination": 6.0, "fiber_length": 688.6727439710183},
        10.0: {"source": 5.0, "destination": 16.0, "fiber_length": 546.8613702582421},
        11.0: {"source": 6.0, "destination": 17.0, "fiber_length": 790.7261094011917},
        12.0: {"source": 7.0, "destination": 26.0, "fiber_length": 1095.95956804919},
        13.0: {"source": 7.0, "destination": 8.0, "fiber_length": 580.7625411994072},
        14.0: {"source": 9.0, "destination": 10.0, "fiber_length": 186.72209514389},
        15.0: {"source": 9.0, "destination": 31.0, "fiber_length": 1482.044136886833},
        16.0: {"source": 10.0, "destination": 17.0, "fiber_length": 496.2580235875224},
        17.0: {"source": 11.0, "destination": 20.0, "fiber_length": 85.61340860642284},
        18.0: {"source": 11.0, "destination": 12.0, "fiber_length": 216.3834396565364},
        19.0: {"source": 12.0, "destination": 13.0, "fiber_length": 70.0045751354164},
        20.0: {"source": 13.0, "destination": 18.0, "fiber_length": 112.9235015085302},
        21.0: {"source": 14.0, "destination": 20.0, "fiber_length": 458.6437394496578},
        22.0: {"source": 14.0, "destination": 21.0, "fiber_length": 1162.804701723245},
        23.0: {"source": 15.0, "destination": 20.0, "fiber_length": 231.7964268611971},
        24.0: {"source": 15.0, "destination": 16.0, "fiber_length": 596.3202865491289},
        25.0: {"source": 18.0, "destination": 19.0, "fiber_length": 20.83436718327859},
        26.0: {"source": 19.0, "destination": 21.0, "fiber_length": 459.728758316418},
        27.0: {"source": 22.0, "destination": 31.0, "fiber_length": 372.4408322573432},
        28.0: {"source": 22.0, "destination": 23.0, "fiber_length": 227.3036100154351},
        29.0: {"source": 24.0, "destination": 25.0, "fiber_length": 796.0775828070355},
        30.0: {"source": 25.0, "destination": 26.0, "fiber_length": 2487.99326220719},
        31.0: {"source": 26.0, "destination": 27.0, "fiber_length": 72.02261095132675},
        32.0: {"source": 26.0, "destination": 30.0, "fiber_length": 544.4600499700148},
        33.0: {"source": 27.0, "destination": 28.0, "fiber_length": 414.6325937752576},
        34.0: {"source": 28.0, "destination": 29.0, "fiber_length": 3017.235249548291},
        35.0: {"source": 29.0, "destination": 30.0, "fiber_length": 3304.814480308357},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
