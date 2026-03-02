def create_ans_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 41.76, 'long': -72.69, 'location': 'Hartford', 'country': 'United States of America (the)'},
    2: {'lat': 40.71, 'long': -74.01, 'location': 'New York City', 'country': 'United States of America (the)'},
    3: {'lat': 41.85, 'long': -87.65, 'location': 'Chicago', 'country': 'United States of America (the)'},
    4: {'lat': 41.5, 'long': -81.7, 'location': 'Cleveland', 'country': 'United States of America (the)'},
    5: {'lat': 36.07, 'long': -79.79, 'location': 'Greensboro', 'country': 'United States of America (the)'},
    6: {'lat': 33.75, 'long': -84.39, 'location': 'Atlanta', 'country': 'United States of America (the)'},
    7: {'lat': 38.9, 'long': -77.04, 'location': 'Washington, D.C.', 'country': 'United States of America (the)'},
    8: {'lat': 38.97, 'long': -77.34, 'location': 'Reston', 'country': 'United States of America (the)'},
    9: {'lat': 32.78, 'long': -96.81, 'location': 'Dallas', 'country': 'United States of America (the)'},
    10: {'lat': 38.63, 'long': -90.2, 'location': 'St. Louis', 'country': 'United States of America (the)'},
    11: {'lat': 47.61, 'long': -122.33, 'location': 'Seattle', 'country': 'United States of America (the)'},
    12: {'lat': 39.74, 'long': -104.98, 'location': 'Denver', 'country': 'United States of America (the)'},
    13: {'lat': 37.77, 'long': -122.42, 'location': 'San Francisco', 'country': 'United States of America (the)'},
    14: {'lat': 37.34, 'long': -121.89, 'location': 'San Jose', 'country': 'United States of America (the)'},
    15: {'lat': 34.05, 'long': -118.24, 'location': 'Los Angeles', 'country': 'United States of America (the)'},
    16: {'lat': 35.08, 'long': -106.65, 'location': 'Albuquerque', 'country': 'United States of America (the)'},
    17: {'lat': 21.31, 'long': -157.86, 'location': 'Honolulu', 'country': 'United States of America (the)'},
    18: {'lat': 29.76, 'long': -95.36, 'location': 'Houston', 'country': 'United States of America (the)'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 240.9995117259535},
    2.0: {'source': 1.0, 'destination': 4.0, 'fiber_length': 1123.590502410244},
    3.0: {'source': 2.0, 'destination': 4.0, 'fiber_length': 975.077114932617},
    4.0: {'source': 2.0, 'destination': 7.0, 'fiber_length': 491.7553096743694},
    5.0: {'source': 2.0, 'destination': 8.0, 'fiber_length': 515.7906472027208},
    6.0: {'source': 3.0, 'destination': 4.0, 'fiber_length': 743.4071200946746},
    7.0: {'source': 3.0, 'destination': 10.0, 'fiber_length': 627.5025611615271},
    8.0: {'source': 3.0, 'destination': 12.0, 'fiber_length': 1843.716434385458},
    9.0: {'source': 5.0, 'destination': 6.0, 'fiber_length': 738.5136638534078},
    10.0: {'source': 5.0, 'destination': 7.0, 'fiber_length': 595.9813603183608},
    11.0: {'source': 6.0, 'destination': 18.0, 'fiber_length': 1500.0},
    12.0: {'source': 7.0, 'destination': 8.0, 'fiber_length': 40.63570747941723},
    13.0: {'source': 8.0, 'destination': 9.0, 'fiber_length': 2349.622138347818},
    14.0: {'source': 8.0, 'destination': 10.0, 'fiber_length': 1500.0},
    15.0: {'source': 9.0, 'destination': 10.0, 'fiber_length': 1323.434150934681},
    16.0: {'source': 9.0, 'destination': 14.0, 'fiber_length': 2913.723191579052},
    17.0: {'source': 9.0, 'destination': 18.0, 'fiber_length': 544.4600499700148},
    18.0: {'source': 11.0, 'destination': 12.0, 'fiber_length': 2051.773895986218},
    19.0: {'source': 11.0, 'destination': 13.0, 'fiber_length': 1500.0},
    20.0: {'source': 12.0, 'destination': 13.0, 'fiber_length': 1906.958816527235},
    21.0: {'source': 13.0, 'destination': 14.0, 'fiber_length': 100.2750862366293},
    22.0: {'source': 13.0, 'destination': 15.0, 'fiber_length': 838.7553033988663},
    23.0: {'source': 15.0, 'destination': 16.0, 'fiber_length': 1500.0},
    24.0: {'source': 16.0, 'destination': 17.0, 'fiber_length': 6483.773338999908},
    25.0: {'source': 16.0, 'destination': 18.0, 'fiber_length': 1515.610552563167},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
