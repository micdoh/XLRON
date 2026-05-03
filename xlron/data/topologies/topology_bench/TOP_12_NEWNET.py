def create_newnet_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {"lat": 47.6038321, "long": -122.330062, "location": "Seattle", "country": "USA"},
        2: {"lat": 45.5202471, "long": -122.674194, "location": "Portland", "country": "USA"},
        3: {"lat": 43.6166163, "long": -116.200886, "location": "Boise", "country": "USA"},
        4: {"lat": 37.3688301, "long": -122.036349, "location": "Sunnyvale", "country": "USA"},
        5: {"lat": 34.0536909, "long": -118.242766, "location": "Los Angeles", "country": "USA"},
        6: {"lat": 40.7596198, "long": -111.886797, "location": "Salt Lake City", "country": "USA"},
        7: {"lat": 32.7174202, "long": -117.162772, "location": "San Diego", "country": "USA"},
        8: {"lat": 39.7392364, "long": -104.984862, "location": "Denver", "country": "USA"},
        9: {"lat": 35.0841034, "long": -106.650985, "location": "Albuquerque", "country": "USA"},
        10: {"lat": 31.7601164, "long": -106.4870404, "location": "El Paso", "country": "USA"},
        11: {"lat": 39.100105, "long": -94.5781416, "location": "Kansas City", "country": "USA"},
        12: {"lat": 36.1563122, "long": -95.9927516, "location": "Tulsa", "country": "USA"},
        13: {"lat": 29.7589382, "long": -95.3676974, "location": "Houston", "country": "USA"},
        14: {"lat": 30.4494155, "long": -91.1869659, "location": "Baton Rouge", "country": "USA"},
        15: {"lat": 41.8755616, "long": -87.6244212, "location": "Chicago", "country": "USA"},
        16: {"lat": 39.7683331, "long": -86.1583502, "location": "Indianapolis", "country": "USA"},
        17: {"lat": 36.1622767, "long": -86.7742984, "location": "Nashville", "country": "USA"},
        18: {"lat": 33.7489924, "long": -84.3902644, "location": "Atlanta", "country": "USA"},
        19: {"lat": 30.3321838, "long": -81.655651, "location": "Jacksonville", "country": "USA"},
        20: {"lat": 35.7803977, "long": -78.6390989, "location": "Raleigh", "country": "USA"},
        21: {"lat": 41.4996574, "long": -81.6936772, "location": "Cleveland", "country": "USA"},
        22: {"lat": 40.4416941, "long": -79.9900861, "location": "Pittsburgh", "country": "USA"},
        23: {"lat": 38.8950368, "long": -77.0365427, "location": "Washington", "country": "USA"},
        24: {"lat": 39.9527237, "long": -75.1635262, "location": "Philadelphia", "country": "USA"},
        25: {"lat": 40.7127281, "long": -74.0060152, "location": "New York", "country": "USA"},
        26: {"lat": 42.3554334, "long": -71.060511, "location": "Boston", "country": "USA"},
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 2.0, "fiber_length": 349.755},
        2.0: {"source": 1.0, "destination": 3.0, "fiber_length": 976.0049999999999},
        3.0: {"source": 2.0, "destination": 4.0, "fiber_length": 1361.925},
        4.0: {"source": 3.0, "destination": 6.0, "fiber_length": 714.885},
        5.0: {"source": 4.0, "destination": 6.0, "fiber_length": 1429.77},
        6.0: {"source": 4.0, "destination": 5.0, "fiber_length": 754.635},
        7.0: {"source": 5.0, "destination": 7.0, "fiber_length": 268.875},
        8.0: {"source": 6.0, "destination": 8.0, "fiber_length": 894.705},
        9.0: {"source": 7.0, "destination": 10.0, "fiber_length": 1500.0},
        10.0: {"source": 8.0, "destination": 9.0, "fiber_length": 807.165},
        11.0: {"source": 8.0, "destination": 11.0, "fiber_length": 1344.375},
        12.0: {"source": 9.0, "destination": 10.0, "fiber_length": 554.88},
        13.0: {"source": 10.0, "destination": 13.0, "fiber_length": 1500.0},
        14.0: {"source": 11.0, "destination": 15.0, "fiber_length": 995.7450000000001},
        15.0: {"source": 11.0, "destination": 12.0, "fiber_length": 525.345},
        16.0: {"source": 12.0, "destination": 13.0, "fiber_length": 1070.61},
        17.0: {"source": 13.0, "destination": 14.0, "fiber_length": 614.115},
        18.0: {"source": 14.0, "destination": 19.0, "fiber_length": 1371.045},
        19.0: {"source": 15.0, "destination": 16.0, "fiber_length": 397.1850000000001},
        20.0: {"source": 15.0, "destination": 21.0, "fiber_length": 741.2249999999999},
        21.0: {"source": 16.0, "destination": 17.0, "fiber_length": 606.885},
        22.0: {"source": 17.0, "destination": 18.0, "fiber_length": 517.875},
        23.0: {"source": 18.0, "destination": 20.0, "fiber_length": 857.595},
        24.0: {"source": 18.0, "destination": 19.0, "fiber_length": 688.59},
        25.0: {"source": 20.0, "destination": 23.0, "fiber_length": 561.255},
        26.0: {"source": 21.0, "destination": 22.0, "fiber_length": 277.785},
        27.0: {"source": 21.0, "destination": 26.0, "fiber_length": 1326.3},
        28.0: {"source": 22.0, "destination": 23.0, "fiber_length": 458.595},
        29.0: {"source": 23.0, "destination": 24.0, "fiber_length": 298.92},
        30.0: {"source": 24.0, "destination": 25.0, "fiber_length": 194.235},
        31.0: {"source": 25.0, "destination": 26.0, "fiber_length": 458.55},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
