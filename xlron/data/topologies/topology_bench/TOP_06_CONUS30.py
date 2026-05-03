def create_conus30_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1.0: {"lat": 33.76, "long": 84.42, "location": nan, "country": nan},
        2.0: {"lat": 39.3, "long": 76.61, "location": nan, "country": nan},
        3.0: {"lat": 42.34, "long": 71.02, "location": nan, "country": nan},
        4.0: {"lat": 41.84, "long": 87.68, "location": nan, "country": nan},
        5.0: {"lat": 35.2, "long": 80.83, "location": nan, "country": nan},
        6.0: {"lat": 41.48, "long": 81.68, "location": nan, "country": nan},
        7.0: {"lat": 32.79, "long": 96.77, "location": nan, "country": nan},
        8.0: {"lat": 39.77, "long": 104.87, "location": nan, "country": nan},
        9.0: {"lat": 31.85, "long": 106.44, "location": nan, "country": nan},
        10.0: {"lat": 29.77, "long": 95.39, "location": nan, "country": nan},
        11.0: {"lat": 39.78, "long": 86.15, "location": nan, "country": nan},
        12.0: {"lat": 39.12, "long": 94.55, "location": nan, "country": nan},
        13.0: {"lat": 34.11, "long": 118.41, "location": nan, "country": nan},
        14.0: {"lat": 36.08, "long": 115.17, "location": nan, "country": nan},
        15.0: {"lat": 25.78, "long": 80.21, "location": nan, "country": nan},
        16.0: {"lat": 38.48, "long": 80.65, "location": nan, "country": nan},
        17.0: {"lat": 30.07, "long": 89.93, "location": nan, "country": nan},
        18.0: {"lat": 40.67, "long": 73.94, "location": nan, "country": nan},
        19.0: {"lat": 40.01, "long": 75.13, "location": nan, "country": nan},
        20.0: {"lat": 33.54, "long": 112.07, "location": nan, "country": nan},
        21.0: {"lat": 38.57, "long": 121.47, "location": nan, "country": nan},
        22.0: {"lat": 40.78, "long": 111.93, "location": nan, "country": nan},
        23.0: {"lat": 29.46, "long": 98.51, "location": nan, "country": nan},
        24.0: {"lat": 37.62, "long": 122.38, "location": nan, "country": nan},
        25.0: {"lat": 37.37, "long": 121.92, "location": nan, "country": nan},
        26.0: {"lat": 38.64, "long": 90.24, "location": nan, "country": nan},
        27.0: {"lat": 47.62, "long": 122.35, "location": nan, "country": nan},
        28.0: {"lat": 27.96, "long": 82.48, "location": nan, "country": nan},
        29.0: {"lat": 30.46, "long": 84.28, "location": nan, "country": nan},
        30.0: {"lat": 38.91, "long": 77.02, "location": nan, "country": nan},
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 5.0, "fiber_length": 548.88},
        2.0: {"source": 1.0, "destination": 10.0, "fiber_length": 1500.0},
        3.0: {"source": 2.0, "destination": 19.0, "fiber_length": 223.92},
        4.0: {"source": 2.0, "destination": 30.0, "fiber_length": 83.955},
        5.0: {"source": 3.0, "destination": 6.0, "fiber_length": 1330.05},
        6.0: {"source": 3.0, "destination": 18.0, "fiber_length": 458.88},
        7.0: {"source": 4.0, "destination": 6.0, "fiber_length": 749.925},
        8.0: {"source": 4.0, "destination": 11.0, "fiber_length": 394.14},
        9.0: {"source": 5.0, "destination": 15.0, "fiber_length": 1500.0},
        10.0: {"source": 5.0, "destination": 30.0, "fiber_length": 799.905},
        11.0: {"source": 7.0, "destination": 10.0, "fiber_length": 540.75},
        12.0: {"source": 7.0, "destination": 12.0, "fiber_length": 1097.415},
        13.0: {"source": 8.0, "destination": 9.0, "fiber_length": 1337.895},
        14.0: {"source": 8.0, "destination": 12.0, "fiber_length": 1332.915},
        15.0: {"source": 8.0, "destination": 22.0, "fiber_length": 913.8000000000001},
        16.0: {"source": 9.0, "destination": 20.0, "fiber_length": 838.875},
        17.0: {"source": 9.0, "destination": 23.0, "fiber_length": 1205.25},
        18.0: {"source": 10.0, "destination": 17.0, "fiber_length": 790.815},
        19.0: {"source": 10.0, "destination": 23.0, "fiber_length": 455.34},
        20.0: {"source": 11.0, "destination": 16.0, "fiber_length": 743.76},
        21.0: {"source": 11.0, "destination": 26.0, "fiber_length": 561.675},
        22.0: {"source": 12.0, "destination": 26.0, "fiber_length": 565.26},
        23.0: {"source": 13.0, "destination": 20.0, "fiber_length": 883.47},
        24.0: {"source": 13.0, "destination": 25.0, "fiber_length": 721.995},
        25.0: {"source": 14.0, "destination": 20.0, "fiber_length": 599.6850000000001},
        26.0: {"source": 14.0, "destination": 22.0, "fiber_length": 890.76},
        27.0: {"source": 15.0, "destination": 28.0, "fiber_length": 496.245},
        28.0: {"source": 16.0, "destination": 30.0, "fiber_length": 477.93},
        29.0: {"source": 17.0, "destination": 29.0, "fiber_length": 816.4499999999999},
        30.0: {"source": 18.0, "destination": 19.0, "fiber_length": 187.095},
        31.0: {"source": 21.0, "destination": 22.0, "fiber_length": 1278.21},
        32.0: {"source": 21.0, "destination": 24.0, "fiber_length": 198.435},
        33.0: {"source": 21.0, "destination": 27.0, "fiber_length": 1500.0},
        34.0: {"source": 22.0, "destination": 27.0, "fiber_length": 1500.0},
        35.0: {"source": 24.0, "destination": 25.0, "fiber_length": 73.785},
        36.0: {"source": 28.0, "destination": 29.0, "fiber_length": 492.465},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
