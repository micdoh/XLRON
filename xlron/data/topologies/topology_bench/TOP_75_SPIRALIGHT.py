def create_spiralight_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {
            "lat": 43.04,
            "long": -87.91,
            "location": "Milwaukee",
            "country": "United States of America (the)",
        },
        2: {
            "lat": 42.27,
            "long": -89.09,
            "location": "Rockford",
            "country": "United States of America (the)",
        },
        3: {
            "lat": 42.68,
            "long": -89.02,
            "location": "Janesville",
            "country": "United States of America (the)",
        },
        4: {
            "lat": 43.01,
            "long": -88.23,
            "location": "Waukesha",
            "country": "United States of America (the)",
        },
        5: {
            "lat": 42.59,
            "long": -87.82,
            "location": "Kenosha",
            "country": "United States of America (the)",
        },
        6: {
            "lat": 43.07,
            "long": -89.4,
            "location": "Madison",
            "country": "United States of America (the)",
        },
        7: {
            "lat": 43.92,
            "long": -90.27,
            "location": "New Lisbon",
            "country": "United States of America (the)",
        },
        8: {
            "lat": 44.98,
            "long": -93.26,
            "location": "Minneapolis",
            "country": "United States of America (the)",
        },
        9: {
            "lat": 44.81,
            "long": -91.5,
            "location": "Eau Claire",
            "country": "United States of America (the)",
        },
        10: {
            "lat": 44.96,
            "long": -89.63,
            "location": "Wausau",
            "country": "United States of America (the)",
        },
        11: {
            "lat": 44.52,
            "long": -88.02,
            "location": "Green Bay",
            "country": "United States of America (the)",
        },
        12: {
            "lat": 44.26,
            "long": -88.42,
            "location": "Appleton",
            "country": "United States of America (the)",
        },
        13: {
            "lat": 44.02,
            "long": -88.54,
            "location": "Oshkosh",
            "country": "United States of America (the)",
        },
        14: {
            "lat": 43.77,
            "long": -88.44,
            "location": "Fond du Lac",
            "country": "United States of America (the)",
        },
        15: {
            "lat": 43.46,
            "long": -88.84,
            "location": "Beaver Dam",
            "country": "United States of America (the)",
        },
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 4.0, "fiber_length": 39.33857400442843},
        2.0: {"source": 1.0, "destination": 5.0, "fiber_length": 75.860016937924},
        3.0: {"source": 2.0, "destination": 3.0, "fiber_length": 68.9249489242323},
        4.0: {"source": 2.0, "destination": 5.0, "fiber_length": 165.2065935025056},
        5.0: {"source": 3.0, "destination": 6.0, "fiber_length": 79.92983480781709},
        6.0: {"source": 4.0, "destination": 6.0, "fiber_length": 142.9780858880039},
        7.0: {"source": 6.0, "destination": 7.0, "fiber_length": 176.5788615413109},
        8.0: {"source": 6.0, "destination": 15.0, "fiber_length": 94.11399976081071},
        9.0: {"source": 7.0, "destination": 9.0, "fiber_length": 208.6732898997296},
        10.0: {"source": 8.0, "destination": 9.0, "fiber_length": 209.8743549933172},
        11.0: {"source": 8.0, "destination": 10.0, "fiber_length": 428.3236075781989},
        12.0: {"source": 10.0, "destination": 11.0, "fiber_length": 204.3696485906607},
        13.0: {"source": 11.0, "destination": 12.0, "fiber_length": 64.44807162855902},
        14.0: {"source": 12.0, "destination": 13.0, "fiber_length": 42.52913763533532},
        15.0: {"source": 13.0, "destination": 14.0, "fiber_length": 43.39577057780664},
        16.0: {"source": 14.0, "destination": 15.0, "fiber_length": 70.75718093948329},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
