def create_hibernia_canada_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {
            "lat": 43.08,
            "long": -79.07,
            "location": "Niagara Falls",
            "country": "United States of America (the)",
        },
        2: {"lat": 46.12, "long": -64.8, "location": "Moncton", "country": "Canada"},
        3: {"lat": 47.37, "long": -68.33, "location": "Edmundston", "country": "Canada"},
        4: {"lat": 46.81, "long": -71.21, "location": "Quebec", "country": "Canada"},
        5: {"lat": 45.51, "long": -73.59, "location": "Montreal", "country": "Canada"},
        6: {"lat": 43.7, "long": -79.42, "location": "Toronto", "country": "Canada"},
        7: {"lat": 55.85, "long": -108.48, "location": "Meadow Lake", "country": "Canada"},
        8: {"lat": 46.28, "long": -63.65, "location": "Kensington", "country": "Canada"},
        9: {"lat": 49.87, "long": -121.44, "location": "Hope", "country": "Canada"},
        10: {"lat": 44.65, "long": -63.57, "location": "Halifax", "country": "Canada"},
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 8.0, "fiber_length": 1584.610892758787},
        2.0: {"source": 1.0, "destination": 9.0, "fiber_length": 4105.62024092363},
        3.0: {"source": 2.0, "destination": 10.0, "fiber_length": 284.3799353856231},
        4.0: {"source": 2.0, "destination": 3.0, "fiber_length": 454.0821377745199},
        5.0: {"source": 3.0, "destination": 4.0, "fiber_length": 340.1067851508351},
        6.0: {"source": 4.0, "destination": 5.0, "fiber_length": 350.1389724135474},
        7.0: {"source": 5.0, "destination": 6.0, "fiber_length": 755.0357670265992},
        8.0: {"source": 5.0, "destination": 8.0, "fiber_length": 1160.215825681341},
        9.0: {"source": 6.0, "destination": 7.0, "fiber_length": 3072.183247999704},
        10.0: {"source": 7.0, "destination": 8.0, "fiber_length": 4053.089425176728},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
