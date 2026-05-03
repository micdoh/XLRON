def create_nsfnet13_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {
            "lat": 29.76,
            "long": -95.36,
            "location": "Houston",
            "country": "United States of America (the)",
        },
        2: {
            "lat": 40.35,
            "long": -74.66,
            "location": "Princeton",
            "country": "United States of America (the)",
        },
        3: {
            "lat": 33.75,
            "long": -84.39,
            "location": "Atlanta",
            "country": "United States of America (the)",
        },
        4: {
            "lat": 40.44,
            "long": -80.0,
            "location": "Pittsburgh",
            "country": "United States of America (the)",
        },
        5: {
            "lat": 42.44,
            "long": -76.5,
            "location": "Ithaca",
            "country": "United States of America (the)",
        },
        6: {
            "lat": 47.61,
            "long": -122.33,
            "location": "Seattle",
            "country": "United States of America (the)",
        },
        7: {
            "lat": 37.44,
            "long": -122.14,
            "location": "Palo Alto",
            "country": "United States of America (the)",
        },
        8: {
            "lat": 32.72,
            "long": -117.16,
            "location": "San Diego",
            "country": "United States of America (the)",
        },
        9: {
            "lat": 40.76,
            "long": -111.89,
            "location": "Salt Lake City",
            "country": "United States of America (the)",
        },
        10: {
            "lat": 40.01,
            "long": -105.27,
            "location": "Boulder",
            "country": "United States of America (the)",
        },
        11: {
            "lat": 40.8,
            "long": -96.67,
            "location": "Lincoln",
            "country": "United States of America (the)",
        },
        12: {
            "lat": 40.12,
            "long": -88.24,
            "location": "Champaign",
            "country": "United States of America (the)",
        },
        13: {
            "lat": 42.28,
            "long": -83.74,
            "location": "Ann Arbor",
            "country": "United States of America (the)",
        },
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 3.0, "fiber_length": 1500.0},
        2.0: {"source": 1.0, "destination": 12.0, "fiber_length": 1651.349618378985},
        3.0: {"source": 1.0, "destination": 8.0, "fiber_length": 2618.380400286071},
        4.0: {"source": 2.0, "destination": 3.0, "fiber_length": 1500.0},
        5.0: {"source": 2.0, "destination": 5.0, "fiber_length": 417.7335921677441},
        6.0: {"source": 4.0, "destination": 13.0, "fiber_length": 559.7411680029991},
        7.0: {"source": 5.0, "destination": 13.0, "fiber_length": 892.4382269692537},
        8.0: {"source": 6.0, "destination": 10.0, "fiber_length": 2004.355311849445},
        9.0: {"source": 6.0, "destination": 7.0, "fiber_length": 1500.0},
        10.0: {"source": 7.0, "destination": 13.0, "fiber_length": 4116.295660779633},
        11.0: {"source": 7.0, "destination": 8.0, "fiber_length": 1039.752315052451},
        12.0: {"source": 9.0, "destination": 10.0, "fiber_length": 850.0906077476434},
        13.0: {"source": 10.0, "destination": 12.0, "fiber_length": 1808.835414803015},
        14.0: {"source": 11.0, "destination": 12.0, "fiber_length": 1075.384518944807},
        15.0: {"source": 12.0, "destination": 13.0, "fiber_length": 669.7219934047032},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
