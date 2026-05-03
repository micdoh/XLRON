def create_nobel_us_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {
            "lat": 37.25,
            "long": -122.07,
            "location": "Saratoga",
            "country": "United States of America (the)",
        },
        2: {"lat": 32.42, "long": -117.08, "location": "Quinta del Cedro", "country": "Mexico"},
        3: {
            "lat": 40.0,
            "long": -105.16,
            "location": "Louisville",
            "country": "United States of America (the)",
        },
        4: {
            "lat": 38.52,
            "long": -77.02,
            "location": "La Plata",
            "country": "United States of America (the)",
        },
        5: {
            "lat": 33.44,
            "long": -84.23,
            "location": "Hampton",
            "country": "United States of America (the)",
        },
        6: {
            "lat": 40.06,
            "long": -88.14,
            "location": "Philo",
            "country": "United States of America (the)",
        },
        7: {
            "lat": 42.16,
            "long": -83.43,
            "location": "Romulus",
            "country": "United States of America (the)",
        },
        8: {
            "lat": 40.47,
            "long": -96.42,
            "location": "Tecumseh",
            "country": "United States of America (the)",
        },
        9: {
            "lat": 40.21,
            "long": -74.39,
            "location": "Vista Center",
            "country": "United States of America (the)",
        },
        10: {
            "lat": 42.26,
            "long": -76.3,
            "location": "Owego",
            "country": "United States of America (the)",
        },
        11: {
            "lat": 40.26,
            "long": -79.58,
            "location": "Youngwood",
            "country": "United States of America (the)",
        },
        12: {
            "lat": 29.45,
            "long": -95.21,
            "location": "Alvin",
            "country": "United States of America (the)",
        },
        13: {
            "lat": 40.39,
            "long": -111.55,
            "location": "Midway",
            "country": "United States of America (the)",
        },
        14: {
            "lat": 47.33,
            "long": -122.24,
            "location": "Auburn",
            "country": "United States of America (the)",
        },
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 2.0, "fiber_length": 1055.897111740372},
        2.0: {"source": 1.0, "destination": 13.0, "fiber_length": 1462.79606747934},
        3.0: {"source": 1.0, "destination": 14.0, "fiber_length": 1500.0},
        4.0: {"source": 2.0, "destination": 12.0, "fiber_length": 2635.085279772903},
        5.0: {"source": 2.0, "destination": 14.0, "fiber_length": 2142.985362652671},
        6.0: {"source": 3.0, "destination": 8.0, "fiber_length": 1115.158750919603},
        7.0: {"source": 3.0, "destination": 12.0, "fiber_length": 1852.650421404539},
        8.0: {"source": 3.0, "destination": 13.0, "fiber_length": 816.5308090688554},
        9.0: {"source": 4.0, "destination": 9.0, "fiber_length": 440.9521122322281},
        10.0: {"source": 4.0, "destination": 10.0, "fiber_length": 630.4661909807334},
        11.0: {"source": 4.0, "destination": 12.0, "fiber_length": 2439.452629208644},
        12.0: {"source": 5.0, "destination": 11.0, "fiber_length": 1295.315980516705},
        13.0: {"source": 5.0, "destination": 12.0, "fiber_length": 1500.0},
        14.0: {"source": 6.0, "destination": 8.0, "fiber_length": 1055.648414012252},
        15.0: {"source": 6.0, "destination": 11.0, "fiber_length": 1091.2330992435},
        16.0: {"source": 6.0, "destination": 14.0, "fiber_length": 3540.969936943001},
        17.0: {"source": 7.0, "destination": 9.0, "fiber_length": 1179.779991047546},
        18.0: {"source": 7.0, "destination": 10.0, "fiber_length": 880.7482376207395},
        19.0: {"source": 7.0, "destination": 13.0, "fiber_length": 2934.397003868838},
        20.0: {"source": 9.0, "destination": 11.0, "fiber_length": 660.7987631283457},
        21.0: {"source": 10.0, "destination": 11.0, "fiber_length": 529.4627293090167},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
