def create_packetexchange_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 25.77, 'long': -80.19, 'location': 'Miami', 'country': 'United States of America (the)'},
    2: {'lat': 51.51, 'long': -0.13, 'location': 'London', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    3: {'lat': 39.04, 'long': -77.49, 'location': 'Ashburn', 'country': 'United States of America (the)'},
    4: {'lat': 40.71, 'long': -74.01, 'location': 'New York City', 'country': 'United States of America (the)'},
    5: {'lat': 48.85, 'long': 2.35, 'location': 'Paris', 'country': 'France'},
    6: {'lat': 45.46, 'long': 9.19, 'location': 'Milano', 'country': 'Italy'},
    7: {'lat': 52.37, 'long': 4.89, 'location': 'Amsterdam', 'country': 'Netherlands (Kingdom of the)'},
    8: {'lat': 50.12, 'long': 8.68, 'location': 'Frankfurt am Main', 'country': 'Germany'},
    9: {'lat': 22.29, 'long': 114.16, 'location': 'Hong Kong', 'country': 'Hong Kong'},
    10: {'lat': 1.29, 'long': 103.85, 'location': 'Singapore', 'country': 'Singapore'},
    11: {'lat': 34.05, 'long': -118.24, 'location': 'Los Angeles', 'country': 'United States of America (the)'},
    12: {'lat': 39.11, 'long': -94.63, 'location': 'Kansas City', 'country': 'United States of America (the)'},
    13: {'lat': 41.85, 'long': -87.65, 'location': 'Chicago', 'country': 'United States of America (the)'},
    14: {'lat': 47.61, 'long': -122.33, 'location': 'Seattle', 'country': 'United States of America (the)'},
    15: {'lat': 37.77, 'long': -122.42, 'location': 'San Francisco', 'country': 'United States of America (the)'},
    16: {'lat': 9.93, 'long': -84.08, 'location': 'San Jose', 'country': 'Costa Rica'},
    17: {'lat': 29.76, 'long': -95.36, 'location': 'Houston', 'country': 'United States of America (the)'},
    18: {'lat': 36.17, 'long': -115.14, 'location': 'Las Vegas', 'country': 'United States of America (the)'},
    19: {'lat': 33.45, 'long': -112.07, 'location': 'Phoenix', 'country': 'United States of America (the)'},
    20: {'lat': 32.78, 'long': -96.81, 'location': 'Dallas', 'country': 'United States of America (the)'},
    21: {'lat': 33.75, 'long': -84.39, 'location': 'Atlanta', 'country': 'United States of America (the)'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 17.0, 'fiber_length': 1944.64480016415},
    2.0: {'source': 1.0, 'destination': 21.0, 'fiber_length': 1462.933348688248},
    3.0: {'source': 2.0, 'destination': 3.0, 'fiber_length': 7396.834895988458},
    4.0: {'source': 2.0, 'destination': 4.0, 'fiber_length': 6963.055714635429},
    5.0: {'source': 2.0, 'destination': 5.0, 'fiber_length': 516.656454570689},
    6.0: {'source': 2.0, 'destination': 7.0, 'fiber_length': 535.6163051270169},
    7.0: {'source': 3.0, 'destination': 13.0, 'fiber_length': 1371.117095352517},
    8.0: {'source': 3.0, 'destination': 21.0, 'fiber_length': 1278.565590677604},
    9.0: {'source': 4.0, 'destination': 5.0, 'fiber_length': 7297.274952474701},
    10.0: {'source': 5.0, 'destination': 8.0, 'fiber_length': 717.5980194901786},
    11.0: {'source': 6.0, 'destination': 8.0, 'fiber_length': 779.3454204877503},
    12.0: {'source': 7.0, 'destination': 8.0, 'fiber_length': 545.2339470871901},
    13.0: {'source': 9.0, 'destination': 10.0, 'fiber_length': 3234.560320425366},
    14.0: {'source': 9.0, 'destination': 15.0, 'fiber_length': 13874.10765324697},
    15.0: {'source': 10.0, 'destination': 11.0, 'fiber_length': 17655.0301181791},
    16.0: {'source': 11.0, 'destination': 19.0, 'fiber_length': 861.3692628497527},
    17.0: {'source': 11.0, 'destination': 16.0, 'fiber_length': 5486.978793082079},
    18.0: {'source': 11.0, 'destination': 18.0, 'fiber_length': 551.2511575012703},
    19.0: {'source': 12.0, 'destination': 20.0, 'fiber_length': 1095.95956804919},
    20.0: {'source': 12.0, 'destination': 13.0, 'fiber_length': 996.0670010192133},
    21.0: {'source': 12.0, 'destination': 14.0, 'fiber_length': 3017.65037849368},
    22.0: {'source': 12.0, 'destination': 15.0, 'fiber_length': 3019.37884190465},
    23.0: {'source': 14.0, 'destination': 15.0, 'fiber_length': 1500.0},
    24.0: {'source': 15.0, 'destination': 16.0, 'fiber_length': 6150.061217370029},
    25.0: {'source': 17.0, 'destination': 20.0, 'fiber_length': 544.4600499700148},
    26.0: {'source': 19.0, 'destination': 20.0, 'fiber_length': 1777.377955221904},
    27.0: {'source': 20.0, 'destination': 21.0, 'fiber_length': 1500.0},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
