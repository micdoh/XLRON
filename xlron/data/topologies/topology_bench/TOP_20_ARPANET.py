def create_arpanet_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {"lat": 45.5202471, "long": -122.674194, "location": "Portland", "country": "USA"},
        2: {"lat": 37.7792588, "long": -122.4193286, "location": "San Francisco", "country": "USA"},
        3: {"lat": 34.0536909, "long": -118.242766, "location": "Los Angeles", "country": "USA"},
        4: {"lat": 40.7596198, "long": -111.886797, "location": "Salt Lake City", "country": "USA"},
        5: {"lat": 32.7174202, "long": -117.162772, "location": "San Diego", "country": "USA"},
        6: {"lat": 39.7392364, "long": -104.984862, "location": "Denver", "country": "USA"},
        7: {"lat": 44.9772995, "long": -93.2654692, "location": "Minneapolis", "country": "USA"},
        8: {"lat": 35.4729886, "long": -97.5170536, "location": "Oklahoma City", "country": "USA"},
        9: {"lat": 29.7589382, "long": -95.3676974, "location": "Houston", "country": "USA"},
        10: {"lat": 39.7990175, "long": -89.6439575, "location": "Springfield", "country": "USA"},
        11: {"lat": 29.9759983, "long": -90.0782127, "location": "New orleans", "country": "USA"},
        12: {"lat": 36.1622767, "long": -86.7742984, "location": "Nashville", "country": "USA"},
        13: {"lat": 33.7489924, "long": -84.3902644, "location": "Atlanta", "country": "USA"},
        14: {"lat": 30.3321838, "long": -81.655651, "location": "Jacksonville", "country": "USA"},
        15: {"lat": 4.099917, "long": -72.9088133, "location": "Columbia", "country": "USA"},
        16: {"lat": 38.8950368, "long": -77.0365427, "location": "Washington", "country": "USA"},
        17: {"lat": 40.4416941, "long": -79.9900861, "location": "Pittsburgh", "country": "USA"},
        18: {"lat": 42.3315509, "long": -83.0466403, "location": "Detroit", "country": "USA"},
        19: {"lat": 40.7127281, "long": -74.0060152, "location": "New york", "country": "USA"},
        20: {"lat": 39.7683331, "long": -86.1583502, "location": "Indianapolis", "country": "USA"},
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 2.0, "fiber_length": 1291.53},
        2.0: {"source": 1.0, "destination": 4.0, "fiber_length": 1500.0},
        3.0: {"source": 1.0, "destination": 7.0, "fiber_length": 2862.65},
        4.0: {"source": 2.0, "destination": 4.0, "fiber_length": 1446.885},
        5.0: {"source": 2.0, "destination": 3.0, "fiber_length": 839.0999999999999},
        6.0: {"source": 4.0, "destination": 6.0, "fiber_length": 894.705},
        7.0: {"source": 3.0, "destination": 6.0, "fiber_length": 1669.725},
        8.0: {"source": 3.0, "destination": 5.0, "fiber_length": 268.875},
        9.0: {"source": 6.0, "destination": 8.0, "fiber_length": 1215.855},
        10.0: {"source": 5.0, "destination": 8.0, "fiber_length": 2289.5},
        11.0: {"source": 5.0, "destination": 9.0, "fiber_length": 2617.825},
        12.0: {"source": 8.0, "destination": 12.0, "fiber_length": 1456.74},
        13.0: {"source": 9.0, "destination": 12.0, "fiber_length": 1500.0},
        14.0: {"source": 9.0, "destination": 10.0, "fiber_length": 1540.0125},
        15.0: {"source": 9.0, "destination": 11.0, "fiber_length": 765.855},
        16.0: {"source": 11.0, "destination": 13.0, "fiber_length": 1022.07},
        17.0: {"source": 11.0, "destination": 14.0, "fiber_length": 1215.885},
        18.0: {"source": 14.0, "destination": 13.0, "fiber_length": 688.59},
        19.0: {"source": 14.0, "destination": 15.0, "fiber_length": 3822.5625},
        20.0: {"source": 15.0, "destination": 16.0, "fiber_length": 4864.3375},
        21.0: {"source": 15.0, "destination": 20.0, "fiber_length": 5229.987499999999},
        22.0: {"source": 13.0, "destination": 20.0, "fiber_length": 1031.34},
        23.0: {"source": 13.0, "destination": 19.0, "fiber_length": 1500.5875},
        24.0: {"source": 12.0, "destination": 16.0, "fiber_length": 1365.465},
        25.0: {"source": 12.0, "destination": 10.0, "fiber_length": 714.24},
        26.0: {"source": 10.0, "destination": 7.0, "fiber_length": 971.865},
        27.0: {"source": 20.0, "destination": 18.0, "fiber_length": 579.5550000000001},
        28.0: {"source": 18.0, "destination": 7.0, "fiber_length": 1308.645},
        29.0: {"source": 18.0, "destination": 17.0, "fiber_length": 495.585},
        30.0: {"source": 17.0, "destination": 19.0, "fiber_length": 759.285},
        31.0: {"source": 17.0, "destination": 16.0, "fiber_length": 458.595},
        32.0: {"source": 19.0, "destination": 16.0, "fiber_length": 492.6},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
