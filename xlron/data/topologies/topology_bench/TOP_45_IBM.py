def create_ibm_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 46.43, 'long': -90.25, 'location': 'Hurley', 'country': 'United States of America (the)'},
    2: {'lat': 41.03, 'long': -73.76, 'location': 'White Plains', 'country': 'United States of America (the)'},
    3: {'lat': 42.33, 'long': -83.05, 'location': 'Detroit', 'country': 'United States of America (the)'},
    4: {'lat': 40.46, 'long': -80.6, 'location': 'Toronto', 'country': 'United States of America (the)'},
    5: {'lat': 39.96, 'long': -83.0, 'location': 'Columbus', 'country': 'United States of America (the)'},
    6: {'lat': 38.98, 'long': -77.1, 'location': 'Bethesda', 'country': 'United States of America (the)'},
    7: {'lat': 40.71, 'long': -74.01, 'location': 'New York City', 'country': 'United States of America (the)'},
    8: {'lat': 39.95, 'long': -75.16, 'location': 'Philadelphia', 'country': 'United States of America (the)'},
    9: {'lat': 42.03, 'long': -88.08, 'location': 'Schaumburg', 'country': 'United States of America (the)'},
    10: {'lat': 38.63, 'long': -90.2, 'location': 'St. Louis', 'country': 'United States of America (the)'},
    11: {'lat': 45.64, 'long': -122.66, 'location': 'Vancouver', 'country': 'United States of America (the)'},
    12: {'lat': 37.77, 'long': -122.42, 'location': 'San Francisco', 'country': 'United States of America (the)'},
    13: {'lat': 34.05, 'long': -118.24, 'location': 'Los Angeles', 'country': 'United States of America (the)'},
    14: {'lat': 33.45, 'long': -112.07, 'location': 'Phoenix', 'country': 'United States of America (the)'},
    15: {'lat': 32.78, 'long': -96.81, 'location': 'Dallas', 'country': 'United States of America (the)'},
    16: {'lat': 33.75, 'long': -84.39, 'location': 'Atlanta', 'country': 'United States of America (the)'},
    17: {'lat': 27.95, 'long': -82.46, 'location': 'Tampa', 'country': 'United States of America (the)'},
    18: {'lat': 41.85, 'long': -87.65, 'location': 'Chicago', 'country': 'United States of America (the)'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 4.0, 'fiber_length': 1500.0},
    2.0: {'source': 1.0, 'destination': 8.0, 'fiber_length': 1770.378337531818},
    3.0: {'source': 2.0, 'destination': 18.0, 'fiber_length': 1500.0},
    4.0: {'source': 2.0, 'destination': 6.0, 'fiber_length': 546.7376201003281},
    5.0: {'source': 2.0, 'destination': 7.0, 'fiber_length': 61.99186833748772},
    6.0: {'source': 3.0, 'destination': 10.0, 'fiber_length': 1096.507405115894},
    7.0: {'source': 3.0, 'destination': 4.0, 'fiber_length': 437.2888728948455},
    8.0: {'source': 3.0, 'destination': 5.0, 'fiber_length': 395.3478240391674},
    9.0: {'source': 5.0, 'destination': 9.0, 'fiber_length': 726.5860063792964},
    10.0: {'source': 5.0, 'destination': 6.0, 'fiber_length': 776.8905067766493},
    11.0: {'source': 5.0, 'destination': 7.0, 'fiber_length': 1149.311160528459},
    12.0: {'source': 6.0, 'destination': 16.0, 'fiber_length': 1310.507269119351},
    13.0: {'source': 7.0, 'destination': 17.0, 'fiber_length': 2018.990891544099},
    14.0: {'source': 7.0, 'destination': 8.0, 'fiber_length': 193.5165266446945},
    15.0: {'source': 9.0, 'destination': 13.0, 'fiber_length': 3461.743578981998},
    16.0: {'source': 10.0, 'destination': 18.0, 'fiber_length': 627.5025611615271},
    17.0: {'source': 11.0, 'destination': 12.0, 'fiber_length': 1312.994463666133},
    18.0: {'source': 12.0, 'destination': 18.0, 'fiber_length': 3729.536934968901},
    19.0: {'source': 12.0, 'destination': 13.0, 'fiber_length': 838.7553033988663},
    20.0: {'source': 13.0, 'destination': 14.0, 'fiber_length': 861.3692628497527},
    21.0: {'source': 14.0, 'destination': 15.0, 'fiber_length': 1777.377955221904},
    22.0: {'source': 15.0, 'destination': 18.0, 'fiber_length': 1614.90805684733},
    23.0: {'source': 15.0, 'destination': 16.0, 'fiber_length': 1500.0},
    24.0: {'source': 16.0, 'destination': 17.0, 'fiber_length': 1006.029022461507},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
