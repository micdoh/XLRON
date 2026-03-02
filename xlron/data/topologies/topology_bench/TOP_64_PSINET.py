def create_psinet_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 42.33, 'long': -83.05, 'location': 'Detroit', 'country': 'United States of America (the)'},
    2: {'lat': 32.54, 'long': -82.9, 'location': 'Dublin', 'country': 'United States of America (the)'},
    3: {'lat': 41.85, 'long': -87.65, 'location': 'Chicago', 'country': 'United States of America (the)'},
    4: {'lat': 41.66, 'long': -83.56, 'location': 'Toledo', 'country': 'United States of America (the)'},
    5: {'lat': 38.9, 'long': -77.04, 'location': 'Washington, D.C.', 'country': 'United States of America (the)'},
    6: {'lat': 39.29, 'long': -76.61, 'location': 'Baltimore', 'country': 'United States of America (the)'},
    7: {'lat': 38.97, 'long': -77.39, 'location': 'Herndon', 'country': 'United States of America (the)'},
    8: {'lat': 33.75, 'long': -84.39, 'location': 'Atlanta', 'country': 'United States of America (the)'},
    9: {'lat': 34.23, 'long': -77.94, 'location': 'Wilmington', 'country': 'United States of America (the)'},
    10: {'lat': 39.95, 'long': -75.16, 'location': 'Philadelphia', 'country': 'United States of America (the)'},
    11: {'lat': 42.36, 'long': -71.06, 'location': 'Boston', 'country': 'United States of America (the)'},
    12: {'lat': 42.61, 'long': -83.15, 'location': 'Troy', 'country': 'United States of America (the)'},
    13: {'lat': 40.71, 'long': -74.01, 'location': 'New York City', 'country': 'United States of America (the)'},
    14: {'lat': 40.74, 'long': -74.17, 'location': 'Newark', 'country': 'United States of America (the)'},
    15: {'lat': 39.11, 'long': -94.63, 'location': 'Kansas City', 'country': 'United States of America (the)'},
    16: {'lat': 38.63, 'long': -90.2, 'location': 'St. Louis', 'country': 'United States of America (the)'},
    17: {'lat': 37.77, 'long': -122.42, 'location': 'San Francisco', 'country': 'United States of America (the)'},
    18: {'lat': 37.35, 'long': -121.96, 'location': 'Santa Clara', 'country': 'United States of America (the)'},
    19: {'lat': 34.05, 'long': -118.24, 'location': 'Los Angeles', 'country': 'United States of America (the)'},
    20: {'lat': 33.45, 'long': -112.07, 'location': 'Phoenix', 'country': 'United States of America (the)'},
    21: {'lat': 30.27, 'long': -97.74, 'location': 'Austin', 'country': 'United States of America (the)'},
    22: {'lat': 32.73, 'long': -97.32, 'location': 'Fort Worth', 'country': 'United States of America (the)'},
    23: {'lat': 32.78, 'long': -96.81, 'location': 'Dallas', 'country': 'United States of America (the)'},
    24: {'lat': 29.76, 'long': -95.36, 'location': 'Houston', 'country': 'United States of America (the)'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 4.0, 'fiber_length': 128.3933603139128},
    2.0: {'source': 2.0, 'destination': 3.0, 'fiber_length': 1500.0},
    3.0: {'source': 2.0, 'destination': 4.0, 'fiber_length': 1500.0},
    4.0: {'source': 2.0, 'destination': 5.0, 'fiber_length': 1324.067201764712},
    5.0: {'source': 3.0, 'destination': 4.0, 'fiber_length': 509.8435626304135},
    6.0: {'source': 3.0, 'destination': 16.0, 'fiber_length': 627.5025611615271},
    7.0: {'source': 5.0, 'destination': 6.0, 'fiber_length': 85.61340860642284},
    8.0: {'source': 6.0, 'destination': 9.0, 'fiber_length': 862.4466435705526},
    9.0: {'source': 7.0, 'destination': 13.0, 'fiber_length': 521.0951719218651},
    10.0: {'source': 7.0, 'destination': 8.0, 'fiber_length': 1280.659967318184},
    11.0: {'source': 8.0, 'destination': 24.0, 'fiber_length': 1500.0},
    12.0: {'source': 9.0, 'destination': 10.0, 'fiber_length': 1023.08605411639},
    13.0: {'source': 10.0, 'destination': 14.0, 'fiber_length': 182.2082908548479},
    14.0: {'source': 11.0, 'destination': 13.0, 'fiber_length': 459.728758316418},
    15.0: {'source': 12.0, 'destination': 13.0, 'fiber_length': 1181.526912522182},
    16.0: {'source': 13.0, 'destination': 14.0, 'fiber_length': 20.83436718327859},
    17.0: {'source': 15.0, 'destination': 20.0, 'fiber_length': 2102.128051806963},
    18.0: {'source': 15.0, 'destination': 23.0, 'fiber_length': 1095.95956804919},
    19.0: {'source': 15.0, 'destination': 16.0, 'fiber_length': 580.7625411994072},
    20.0: {'source': 17.0, 'destination': 18.0, 'fiber_length': 92.77121696685674},
    21.0: {'source': 17.0, 'destination': 19.0, 'fiber_length': 838.7553033988663},
    22.0: {'source': 19.0, 'destination': 20.0, 'fiber_length': 861.3692628497527},
    23.0: {'source': 21.0, 'destination': 24.0, 'fiber_length': 354.0915074929874},
    24.0: {'source': 22.0, 'destination': 23.0, 'fiber_length': 72.02261095132675},
    25.0: {'source': 23.0, 'destination': 24.0, 'fiber_length': 544.4600499700148},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
