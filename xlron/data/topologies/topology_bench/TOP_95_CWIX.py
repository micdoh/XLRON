def create_cwix_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 39.95, 'long': -75.16, 'location': 'Philadelphia', 'country': 'United States of America (the)'},
    2: {'lat': 40.44, 'long': -80.0, 'location': 'Pittsburgh', 'country': 'United States of America (the)'},
    3: {'lat': 37.55, 'long': -77.46, 'location': 'Richmond', 'country': 'United States of America (the)'},
    4: {'lat': 39.29, 'long': -76.61, 'location': 'Baltimore', 'country': 'United States of America (the)'},
    5: {'lat': 42.36, 'long': -71.06, 'location': 'Boston', 'country': 'United States of America (the)'},
    6: {'lat': 32.78, 'long': -96.81, 'location': 'Dallas', 'country': 'United States of America (the)'},
    7: {'lat': 40.71, 'long': -74.01, 'location': 'New York City', 'country': 'United States of America (the)'},
    8: {'lat': 41.76, 'long': -72.69, 'location': 'Hartford', 'country': 'United States of America (the)'},
    9: {'lat': 29.76, 'long': -95.36, 'location': 'Houston', 'country': 'United States of America (the)'},
    10: {'lat': 33.75, 'long': -84.39, 'location': 'Atlanta', 'country': 'United States of America (the)'},
    11: {'lat': 26.12, 'long': -80.14, 'location': 'Fort Lauderdale', 'country': 'United States of America (the)'},
    12: {'lat': 28.54, 'long': -81.38, 'location': 'Orlando', 'country': 'United States of America (the)'},
    13: {'lat': 30.33, 'long': -81.66, 'location': 'Jacksonville', 'country': 'United States of America (the)'},
    14: {'lat': 35.77, 'long': -78.64, 'location': 'Raleigh', 'country': 'United States of America (the)'},
    15: {'lat': 27.95, 'long': -82.46, 'location': 'Tampa', 'country': 'United States of America (the)'},
    16: {'lat': 25.77, 'long': -80.19, 'location': 'Miami', 'country': 'United States of America (the)'},
    17: {'lat': 41.08, 'long': -81.52, 'location': 'Akron', 'country': 'United States of America (the)'},
    18: {'lat': 37.77, 'long': -122.42, 'location': 'San Francisco', 'country': 'United States of America (the)'},
    19: {'lat': 34.05, 'long': -118.24, 'location': 'Los Angeles', 'country': 'United States of America (the)'},
    20: {'lat': 39.74, 'long': -104.98, 'location': 'Denver', 'country': 'United States of America (the)'},
    21: {'lat': 39.11, 'long': -94.63, 'location': 'Kansas City', 'country': 'United States of America (the)'},
    22: {'lat': 41.85, 'long': -87.65, 'location': 'Chicago', 'country': 'United States of America (the)'},
    23: {'lat': 41.5, 'long': -81.7, 'location': 'Cleveland', 'country': 'United States of America (the)'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 621.9499312218177},
    2.0: {'source': 1.0, 'destination': 4.0, 'fiber_length': 216.3834396565364},
    3.0: {'source': 2.0, 'destination': 7.0, 'fiber_length': 760.0480661195877},
    4.0: {'source': 2.0, 'destination': 4.0, 'fiber_length': 474.4497917256419},
    5.0: {'source': 2.0, 'destination': 17.0, 'fiber_length': 219.7025529013169},
    6.0: {'source': 3.0, 'destination': 4.0, 'fiber_length': 310.7443227756991},
    7.0: {'source': 3.0, 'destination': 14.0, 'fiber_length': 336.2521540699969},
    8.0: {'source': 4.0, 'destination': 10.0, 'fiber_length': 1392.312707704519},
    9.0: {'source': 5.0, 'destination': 7.0, 'fiber_length': 459.728758316418},
    10.0: {'source': 5.0, 'destination': 8.0, 'fiber_length': 225.2903452207617},
    11.0: {'source': 6.0, 'destination': 9.0, 'fiber_length': 544.4600499700148},
    12.0: {'source': 6.0, 'destination': 10.0, 'fiber_length': 1500.0},
    13.0: {'source': 6.0, 'destination': 21.0, 'fiber_length': 1095.95956804919},
    14.0: {'source': 6.0, 'destination': 19.0, 'fiber_length': 2487.99326220719},
    15.0: {'source': 7.0, 'destination': 22.0, 'fiber_length': 1500.0},
    16.0: {'source': 7.0, 'destination': 8.0, 'fiber_length': 240.9995117259535},
    17.0: {'source': 7.0, 'destination': 18.0, 'fiber_length': 5161.319559832027},
    18.0: {'source': 10.0, 'destination': 15.0, 'fiber_length': 1006.029022461507},
    19.0: {'source': 11.0, 'destination': 15.0, 'fiber_length': 460.3747720207242},
    20.0: {'source': 11.0, 'destination': 16.0, 'fiber_length': 58.85702770663077},
    21.0: {'source': 11.0, 'destination': 12.0, 'fiber_length': 443.4783482522398},
    22.0: {'source': 12.0, 'destination': 13.0, 'fiber_length': 301.3157125461099},
    23.0: {'source': 13.0, 'destination': 14.0, 'fiber_length': 1000.614675789564},
    24.0: {'source': 17.0, 'destination': 22.0, 'fiber_length': 776.6863956572226},
    25.0: {'source': 17.0, 'destination': 23.0, 'fiber_length': 73.59530040173104},
    26.0: {'source': 18.0, 'destination': 20.0, 'fiber_length': 1906.958816527235},
    27.0: {'source': 18.0, 'destination': 19.0, 'fiber_length': 838.7553033988663},
    28.0: {'source': 20.0, 'destination': 21.0, 'fiber_length': 1336.875843009606},
    29.0: {'source': 21.0, 'destination': 22.0, 'fiber_length': 996.0670010192133},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
