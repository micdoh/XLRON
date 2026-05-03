def create_integra_graph():
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
            "lat": 32.78,
            "long": -96.81,
            "location": "Dallas",
            "country": "United States of America (the)",
        },
        3: {
            "lat": 45.56,
            "long": -94.16,
            "location": "Saint Cloud",
            "country": "United States of America (the)",
        },
        4: {
            "lat": 44.98,
            "long": -93.26,
            "location": "Minneapolis",
            "country": "United States of America (the)",
        },
        5: {
            "lat": 34.05,
            "long": -118.24,
            "location": "Los Angeles",
            "country": "United States of America (the)",
        },
        6: {
            "lat": 37.35,
            "long": -121.96,
            "location": "Santa Clara",
            "country": "United States of America (the)",
        },
        7: {
            "lat": 36.17,
            "long": -115.14,
            "location": "Las Vegas",
            "country": "United States of America (the)",
        },
        8: {
            "lat": 33.45,
            "long": -112.07,
            "location": "Phoenix",
            "country": "United States of America (the)",
        },
        9: {
            "lat": 38.58,
            "long": -121.49,
            "location": "Sacramento",
            "country": "United States of America (the)",
        },
        10: {
            "lat": 38.44,
            "long": -122.71,
            "location": "Santa Rosa",
            "country": "United States of America (the)",
        },
        11: {
            "lat": 44.94,
            "long": -123.04,
            "location": "Salem",
            "country": "United States of America (the)",
        },
        12: {
            "lat": 44.06,
            "long": -121.32,
            "location": "Bend",
            "country": "United States of America (the)",
        },
        13: {
            "lat": 44.05,
            "long": -123.09,
            "location": "Eugene",
            "country": "United States of America (the)",
        },
        14: {
            "lat": 39.53,
            "long": -119.81,
            "location": "Reno",
            "country": "United States of America (the)",
        },
        15: {
            "lat": 38.59,
            "long": -121.3,
            "location": "Rancho Cordova",
            "country": "United States of America (the)",
        },
        16: {
            "lat": 40.71,
            "long": -74.01,
            "location": "New York City",
            "country": "United States of America (the)",
        },
        17: {
            "lat": 39.04,
            "long": -77.49,
            "location": "Ashburn",
            "country": "United States of America (the)",
        },
        18: {
            "lat": 47.66,
            "long": -117.43,
            "location": "Spokane",
            "country": "United States of America (the)",
        },
        19: {
            "lat": 46.88,
            "long": -96.79,
            "location": "Fargo",
            "country": "United States of America (the)",
        },
        20: {
            "lat": 45.78,
            "long": -108.5,
            "location": "Billings",
            "country": "United States of America (the)",
        },
        21: {
            "lat": 39.74,
            "long": -104.98,
            "location": "Denver",
            "country": "United States of America (the)",
        },
        22: {
            "lat": 41.22,
            "long": -111.97,
            "location": "Ogden",
            "country": "United States of America (the)",
        },
        23: {
            "lat": 40.3,
            "long": -111.69,
            "location": "Orem",
            "country": "United States of America (the)",
        },
        24: {
            "lat": 40.76,
            "long": -111.89,
            "location": "Salt Lake City",
            "country": "United States of America (the)",
        },
        25: {
            "lat": 43.61,
            "long": -116.2,
            "location": "Boise",
            "country": "United States of America (the)",
        },
        26: {
            "lat": 45.52,
            "long": -122.68,
            "location": "Portland",
            "country": "United States of America (the)",
        },
        27: {
            "lat": 47.61,
            "long": -122.33,
            "location": "Seattle",
            "country": "United States of America (the)",
        },
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 17.0, "fiber_length": 1371.117095352517},
        2.0: {"source": 1.0, "destination": 4.0, "fiber_length": 856.6967858075002},
        3.0: {"source": 1.0, "destination": 21.0, "fiber_length": 1843.716434385458},
        4.0: {"source": 2.0, "destination": 21.0, "fiber_length": 1500.0},
        5.0: {"source": 2.0, "destination": 8.0, "fiber_length": 1777.377955221904},
        6.0: {"source": 3.0, "destination": 19.0, "fiber_length": 374.9189023995028},
        7.0: {"source": 3.0, "destination": 4.0, "fiber_length": 143.2439767771629},
        8.0: {"source": 4.0, "destination": 19.0, "fiber_length": 517.7263785550638},
        9.0: {"source": 4.0, "destination": 27.0, "fiber_length": 2798.902600946488},
        10.0: {"source": 4.0, "destination": 16.0, "fiber_length": 2043.907199971668},
        11.0: {"source": 5.0, "destination": 6.0, "fiber_length": 746.0789546600618},
        12.0: {"source": 5.0, "destination": 7.0, "fiber_length": 551.2511575012703},
        13.0: {"source": 5.0, "destination": 8.0, "fiber_length": 861.3692628497527},
        14.0: {"source": 6.0, "destination": 9.0, "fiber_length": 214.2607789089177},
        15.0: {"source": 7.0, "destination": 24.0, "fiber_length": 875.1907174327788},
        16.0: {"source": 7.0, "destination": 8.0, "fiber_length": 618.4493777535091},
        17.0: {"source": 9.0, "destination": 10.0, "fiber_length": 160.9300921413741},
        18.0: {"source": 9.0, "destination": 14.0, "fiber_length": 269.1664316225356},
        19.0: {"source": 9.0, "destination": 15.0, "fiber_length": 24.82807554679378},
        20.0: {"source": 9.0, "destination": 26.0, "fiber_length": 1166.845178904244},
        21.0: {"source": 11.0, "destination": 26.0, "fiber_length": 105.5780583789585},
        22.0: {"source": 11.0, "destination": 13.0, "fiber_length": 148.5643640399954},
        23.0: {"source": 12.0, "destination": 26.0, "fiber_length": 291.9085042218529},
        24.0: {"source": 12.0, "destination": 27.0, "fiber_length": 603.6183344061474},
        25.0: {"source": 14.0, "destination": 24.0, "fiber_length": 1030.031392577858},
        26.0: {"source": 16.0, "destination": 17.0, "fiber_length": 525.3065372949657},
        27.0: {"source": 18.0, "destination": 27.0, "fiber_length": 550.6983275646231},
        28.0: {"source": 18.0, "destination": 20.0, "fiber_length": 1067.463439813683},
        29.0: {"source": 20.0, "destination": 21.0, "fiber_length": 1095.480395458758},
        30.0: {"source": 21.0, "destination": 22.0, "fiber_length": 920.2225536447613},
        31.0: {"source": 22.0, "destination": 23.0, "fiber_length": 157.4733440312334},
        32.0: {"source": 22.0, "destination": 24.0, "fiber_length": 77.38275291458565},
        33.0: {"source": 23.0, "destination": 24.0, "fiber_length": 80.80529193589061},
        34.0: {"source": 24.0, "destination": 25.0, "fiber_length": 713.7523611067775},
        35.0: {"source": 25.0, "destination": 26.0, "fiber_length": 833.0136408211406},
        36.0: {"source": 26.0, "destination": 27.0, "fiber_length": 350.8979570264834},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
