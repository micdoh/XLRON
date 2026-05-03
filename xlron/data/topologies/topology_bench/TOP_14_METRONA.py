def create_metrona_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {"lat": 50.3714122, "long": -4.1424451, "location": "Plymouth", "country": nan},
        2: {"lat": 50.6190962, "long": -3.4146801, "location": "Exmouth", "country": nan},
        3: {"lat": 51.0147895, "long": -3.1029086, "location": "Taunton", "country": nan},
        4: {"lat": 51.4538022, "long": -2.5972985, "location": "Bristol", "country": nan},
        5: {"lat": 50.9025349, "long": -1.404189, "location": "Southampton", "country": nan},
        6: {"lat": 50.8214626, "long": -0.1400561, "location": "Brighton", "country": nan},
        7: {"lat": 51.6195955, "long": -3.9459248, "location": "Swansea", "country": nan},
        8: {"lat": 51.4816546, "long": -3.1791934, "location": "Cardiff", "country": nan},
        9: {"lat": 51.4564242, "long": -0.9700664, "location": "Reading", "country": nan},
        10: {"lat": 51.8896903, "long": 0.8994651, "location": "Colchester", "country": nan},
        11: {"lat": 52.6285576, "long": 1.2923954, "location": "Norwich", "country": nan},
        12: {"lat": 51.5074456, "long": -0.1277653, "location": "London", "country": nan},
        13: {"lat": 40.0271453, "long": -78.5237447, "location": "Bedford", "country": nan},
        14: {
            "lat": 52.23433665,
            "long": -0.902807276818582,
            "location": "Northampton",
            "country": nan,
        },
        15: {"lat": 52.5725769, "long": -0.2427336, "location": "Peterborough", "country": nan},
        16: {"lat": 52.4081812, "long": -1.510477, "location": "Coventry", "country": nan},
        17: {"lat": 52.4552224, "long": -1.1997815, "location": "Lutterworth", "country": nan},
        18: {"lat": 52.6362, "long": -1.1331969, "location": "Leicester", "country": nan},
        19: {"lat": 53.3806626, "long": -1.4702278, "location": "Sheffield", "country": nan},
        20: {"lat": 52.4796992, "long": -1.9026911, "location": "Birmingham", "country": nan},
        21: {"lat": 52.1911849, "long": -2.2206585, "location": "Worcester", "country": nan},
        22: {"lat": 53.0162014, "long": -2.1812607, "location": "Stoke-on-Trent", "country": nan},
        23: {"lat": 53.4794892, "long": -2.2451148, "location": "Manchester", "country": nan},
        24: {"lat": 53.4071991, "long": -2.99168, "location": "Liverpool", "country": nan},
        25: {"lat": 53.7593363, "long": -2.6992717, "location": "Preston", "country": nan},
        26: {"lat": 53.7944229, "long": -1.7519186, "location": "Bradford", "country": nan},
        27: {"lat": 53.7974185, "long": -1.5437941, "location": "Leeds", "country": nan},
        28: {"lat": 54.5760419, "long": -1.2344047, "location": "Middlesbrough", "country": nan},
        29: {"lat": 54.9058512, "long": -1.3828727, "location": "Sunderland", "country": nan},
        30: {"lat": 54.9738474, "long": -1.6131572, "location": "Newcastle", "country": nan},
        31: {"lat": 55.0691397, "long": -3.6107936, "location": "Dumfries", "country": nan},
        32: {"lat": 55.9533456, "long": -3.1883749, "location": "Edinburgh", "country": nan},
        33: {"lat": 55.861155, "long": -4.2501687, "location": "Glasgow", "country": nan},
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 2.0, "fiber_length": 87.57000000000001},
        2.0: {"source": 1.0, "destination": 5.0, "fiber_length": 302.895},
        3.0: {"source": 2.0, "destination": 3.0, "fiber_length": 73.725},
        4.0: {"source": 3.0, "destination": 4.0, "fiber_length": 90.27},
        5.0: {"source": 4.0, "destination": 9.0, "fiber_length": 169.125},
        6.0: {"source": 4.0, "destination": 21.0, "fiber_length": 128.97},
        7.0: {"source": 5.0, "destination": 6.0, "fiber_length": 133.77},
        8.0: {"source": 6.0, "destination": 12.0, "fiber_length": 114.42},
        9.0: {"source": 7.0, "destination": 8.0, "fiber_length": 82.785},
        10.0: {"source": 7.0, "destination": 21.0, "fiber_length": 201.51},
        11.0: {"source": 8.0, "destination": 21.0, "fiber_length": 154.155},
        12.0: {"source": 9.0, "destination": 21.0, "fiber_length": 177.87},
        13.0: {"source": 9.0, "destination": 12.0, "fiber_length": 87.9},
        14.0: {"source": 10.0, "destination": 12.0, "fiber_length": 123.855},
        15.0: {"source": 10.0, "destination": 11.0, "fiber_length": 129.6},
        16.0: {"source": 11.0, "destination": 15.0, "fiber_length": 155.79},
        17.0: {"source": 12.0, "destination": 13.0, "fiber_length": 7392.174999999999},
        18.0: {"source": 13.0, "destination": 15.0, "fiber_length": 7332.212500000001},
        19.0: {"source": 13.0, "destination": 14.0, "fiber_length": 7295.175},
        20.0: {"source": 14.0, "destination": 16.0, "fiber_length": 68.4},
        21.0: {"source": 16.0, "destination": 17.0, "fiber_length": 32.55},
        22.0: {"source": 16.0, "destination": 20.0, "fiber_length": 41.625},
        23.0: {"source": 17.0, "destination": 18.0, "fiber_length": 30.93},
        24.0: {"source": 18.0, "destination": 19.0, "fiber_length": 128.7},
        25.0: {"source": 19.0, "destination": 27.0, "fiber_length": 69.885},
        26.0: {"source": 20.0, "destination": 21.0, "fiber_length": 58.02},
        27.0: {"source": 20.0, "destination": 22.0, "fiber_length": 93.795},
        28.0: {"source": 22.0, "destination": 23.0, "fiber_length": 77.535},
        29.0: {"source": 23.0, "destination": 24.0, "fiber_length": 75.135},
        30.0: {"source": 23.0, "destination": 26.0, "fiber_length": 71.685},
        31.0: {"source": 23.0, "destination": 25.0, "fiber_length": 64.785},
        32.0: {"source": 24.0, "destination": 25.0, "fiber_length": 65.475},
        33.0: {"source": 25.0, "destination": 31.0, "fiber_length": 235.695},
        34.0: {"source": 26.0, "destination": 27.0, "fiber_length": 20.505},
        35.0: {"source": 26.0, "destination": 28.0, "fiber_length": 139.815},
        36.0: {"source": 28.0, "destination": 29.0, "fiber_length": 56.835},
        37.0: {"source": 29.0, "destination": 30.0, "fiber_length": 24.81},
        38.0: {"source": 30.0, "destination": 31.0, "fiber_length": 191.655},
        39.0: {"source": 30.0, "destination": 32.0, "fiber_length": 221.07},
        40.0: {"source": 32.0, "destination": 33.0, "fiber_length": 100.455},
        41.0: {"source": 33.0, "destination": 31.0, "fiber_length": 145.275},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
