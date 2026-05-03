def create_oxford_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {
            "lat": 43.07,
            "long": -70.76,
            "location": "Portsmouth",
            "country": "United States of America (the)",
        },
        2: {
            "lat": 43.2,
            "long": -70.87,
            "location": "Dover",
            "country": "United States of America (the)",
        },
        3: {
            "lat": 43.49,
            "long": -70.45,
            "location": "Biddeford",
            "country": "United States of America (the)",
        },
        4: {
            "lat": 43.44,
            "long": -70.77,
            "location": "Sanford (historical)",
            "country": "United States of America (the)",
        },
        5: {
            "lat": 42.36,
            "long": -71.06,
            "location": "Boston",
            "country": "United States of America (the)",
        },
        6: {
            "lat": 42.28,
            "long": -71.42,
            "location": "Framingham",
            "country": "United States of America (the)",
        },
        7: {
            "lat": 43.0,
            "long": -71.45,
            "location": "Manchester",
            "country": "United States of America (the)",
        },
        8: {
            "lat": 42.93,
            "long": -72.28,
            "location": "Keene",
            "country": "United States of America (the)",
        },
        9: {
            "lat": 42.26,
            "long": -71.8,
            "location": "Worcester",
            "country": "United States of America (the)",
        },
        10: {
            "lat": 42.1,
            "long": -72.59,
            "location": "Springfield",
            "country": "United States of America (the)",
        },
        11: {
            "lat": 44.8,
            "long": -68.78,
            "location": "Bangor",
            "country": "United States of America (the)",
        },
        12: {
            "lat": 43.66,
            "long": -70.26,
            "location": "Portland",
            "country": "United States of America (the)",
        },
        13: {
            "lat": 44.26,
            "long": -70.26,
            "location": "Turner",
            "country": "United States of America (the)",
        },
        14: {
            "lat": 44.4,
            "long": -70.79,
            "location": "Bethel",
            "country": "United States of America (the)",
        },
        15: {
            "lat": 44.21,
            "long": -70.54,
            "location": "Norway",
            "country": "United States of America (the)",
        },
        16: {
            "lat": 43.74,
            "long": -70.55,
            "location": "Buxton",
            "country": "United States of America (the)",
        },
        17: {
            "lat": 44.29,
            "long": -70.37,
            "location": "Buckfield",
            "country": "United States of America (the)",
        },
        18: {
            "lat": 44.31,
            "long": -69.78,
            "location": "Augusta",
            "country": "United States of America (the)",
        },
        19: {
            "lat": 44.1,
            "long": -70.21,
            "location": "Lewiston",
            "country": "United States of America (the)",
        },
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 12.0, "fiber_length": 115.5842950114381},
        2.0: {"source": 1.0, "destination": 4.0, "fiber_length": 61.72513877291561},
        3.0: {"source": 1.0, "destination": 7.0, "fiber_length": 84.92727922577953},
        4.0: {"source": 2.0, "destination": 12.0, "fiber_length": 106.5170591971861},
        5.0: {"source": 2.0, "destination": 8.0, "fiber_length": 177.6169985687207},
        6.0: {"source": 3.0, "destination": 12.0, "fiber_length": 36.48425251511242},
        7.0: {"source": 3.0, "destination": 4.0, "fiber_length": 39.62573932255859},
        8.0: {"source": 5.0, "destination": 6.0, "fiber_length": 46.35899387981959},
        9.0: {"source": 5.0, "destination": 7.0, "fiber_length": 116.9687891029035},
        10.0: {"source": 6.0, "destination": 9.0, "fiber_length": 47.01940919936226},
        11.0: {"source": 8.0, "destination": 10.0, "fiber_length": 143.587760082976},
        12.0: {"source": 9.0, "destination": 10.0, "fiber_length": 101.2244765261993},
        13.0: {"source": 11.0, "destination": 19.0, "fiber_length": 206.4456940218038},
        14.0: {"source": 11.0, "destination": 18.0, "fiber_length": 144.2390619276744},
        15.0: {"source": 12.0, "destination": 19.0, "fiber_length": 73.6344166653572},
        16.0: {"source": 12.0, "destination": 16.0, "fiber_length": 37.42898611293066},
        17.0: {"source": 13.0, "destination": 18.0, "fiber_length": 57.91672625596681},
        18.0: {"source": 13.0, "destination": 14.0, "fiber_length": 67.40841461515036},
        19.0: {"source": 14.0, "destination": 15.0, "fiber_length": 43.52863261777786},
        20.0: {"source": 15.0, "destination": 17.0, "fiber_length": 24.30150946072797},
        21.0: {"source": 15.0, "destination": 16.0, "fiber_length": 78.40161184957121},
        22.0: {"source": 16.0, "destination": 17.0, "fiber_length": 94.24233482276554},
        23.0: {"source": 17.0, "destination": 18.0, "fiber_length": 70.50834274957555},
        24.0: {"source": 18.0, "destination": 19.0, "fiber_length": 62.21034204625666},
        25.0: {"source": 18.0, "destination": 18.0, "fiber_length": 0.0},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
