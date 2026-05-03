def create_lambdarail_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {"lat": 47.60621, "long": -122.33207, "location": nan, "country": "USA"},
        2: {"lat": 37.368888888889, "long": -122.03694444444, "location": nan, "country": "USA"},
        3: {"lat": 34.05223, "long": -118.24368, "location": nan, "country": "USA"},
        4: {"lat": 39.733, "long": -104.99, "location": nan, "country": "USA"},
        5: {"lat": 35.116666666667, "long": -106.61666666667, "location": nan, "country": "USA"},
        6: {"lat": 31.759166666667, "long": -106.48861111111, "location": nan, "country": "USA"},
        7: {"lat": 39.05, "long": -94.583333333333, "location": nan, "country": "USA"},
        8: {"lat": 36.131388888889, "long": -95.937222222222, "location": nan, "country": "USA"},
        9: {"lat": 29.762777777778, "long": -95.383055555556, "location": nan, "country": "USA"},
        10: {"lat": 41.85003, "long": -87.65005, "location": nan, "country": "USA"},
        11: {"lat": 30.4475, "long": -91.178611111111, "location": nan, "country": "USA"},
        12: {"lat": 41.482222222222, "long": -81.669722222222, "location": nan, "country": "USA"},
        13: {"lat": 33.756944444444, "long": -84.390277777778, "location": nan, "country": "USA"},
        14: {"lat": 33.815766, "long": -85.760467, "location": nan, "country": "USA"},
        15: {"lat": 35.78, "long": -78.64, "location": nan, "country": "USA"},
        16: {"lat": 40.49, "long": -88.97, "location": nan, "country": "USA"},
        17: {"lat": 40.71, "long": -74.03, "location": nan, "country": "USA"},
        18: {"lat": 42.375, "long": -71.106111111111, "location": nan, "country": "USA"},
        19: {"lat": 40.4416666666667, "long": -80.0, "location": nan, "country": "USA"},
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 2.0, "fiber_length": 1500.0},
        2.0: {"source": 1.0, "destination": 4.0, "fiber_length": 2051.475},
        3.0: {"source": 2.0, "destination": 3.0, "fiber_length": 754.8},
        4.0: {"source": 4.0, "destination": 5.0, "fiber_length": 799.5},
        5.0: {"source": 5.0, "destination": 6.0, "fiber_length": 560.295},
        6.0: {"source": 3.0, "destination": 6.0, "fiber_length": 1500.0},
        7.0: {"source": 4.0, "destination": 7.0, "fiber_length": 1345.5},
        8.0: {"source": 7.0, "destination": 8.0, "fiber_length": 518.625},
        9.0: {"source": 8.0, "destination": 9.0, "fiber_length": 1065.06},
        10.0: {"source": 9.0, "destination": 6.0, "fiber_length": 1500.0},
        11.0: {"source": 9.0, "destination": 11.0, "fiber_length": 617.295},
        12.0: {"source": 11.0, "destination": 14.0, "fiber_length": 949.095},
        13.0: {"source": 14.0, "destination": 13.0, "fiber_length": 190.2},
        14.0: {"source": 13.0, "destination": 15.0, "fiber_length": 856.905},
        15.0: {"source": 15.0, "destination": 16.0, "fiber_length": 1500.0},
        16.0: {"source": 16.0, "destination": 19.0, "fiber_length": 1137.78},
        17.0: {"source": 19.0, "destination": 12.0, "fiber_length": 272.655},
        18.0: {"source": 12.0, "destination": 10.0, "fiber_length": 747.5250000000001},
        19.0: {"source": 10.0, "destination": 7.0, "fiber_length": 995.79},
        20.0: {"source": 16.0, "destination": 17.0, "fiber_length": 1575.075},
        21.0: {"source": 17.0, "destination": 12.0, "fiber_length": 968.5500000000001},
        22.0: {"source": 17.0, "destination": 18.0, "fiber_length": 458.595},
        23.0: {"source": 18.0, "destination": 10.0, "fiber_length": 1704.7125},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
