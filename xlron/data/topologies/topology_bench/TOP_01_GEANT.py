def create_geant_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 48.2091, 'long': 16.3729, 'location': 'Vienna', 'country': 'Austria'},
    2: {'lat': 50.8469, 'long': 4.3518, 'location': 'Brussels', 'country': 'Belgium'},
    3: {'lat': 46.2038, 'long': 6.1399, 'location': 'Geneve', 'country': 'Switzerland'},
    4: {'lat': 50.0785, 'long': 14.4423, 'location': 'Karlin', 'country': 'Czechia'},
    5: {'lat': 50.1122, 'long': 8.6842, 'location': 'Frankfurt am Main', 'country': 'Germany'},
    6: {'lat': 40.4167, 'long': -3.7033, 'location': 'Madrid', 'country': 'Spain'},
    7: {'lat': 48.8566, 'long': 2.351, 'location': 'Paris', 'country': 'France'},
    8: {'lat': 37.9778, 'long': 23.5808, 'location': 'Perama', 'country': 'Greece'},
    9: {'lat': 45.8071, 'long': 15.9644, 'location': 'Zagreb - Centar', 'country': 'Croatia'},
    10: {'lat': 47.4976, 'long': 19.0936, 'location': 'Budapest VIII. keruelet', 'country': 'Hungary'},
    11: {'lat': 53.3416, 'long': -6.2573, 'location': 'Dublin', 'country': 'Ireland'},
    12: {'lat': 32.0714, 'long': 34.8097, 'location': 'Giv`atayim', 'country': 'Israel'},
    13: {'lat': 45.4642, 'long': 9.19, 'location': 'Milano', 'country': 'Italy'},
    14: {'lat': 49.6112, 'long': 6.1296, 'location': 'Luxembourg', 'country': 'Luxembourg'},
    15: {'lat': 52.3236, 'long': 4.9407, 'location': 'Duivendrecht', 'country': 'Netherlands (Kingdom of the)'},
    16: {'lat': 40.6698, 'long': -73.94384, 'location': 'Brooklyn', 'country': 'United States of America (the)'},
    17: {'lat': 52.3963, 'long': 16.8874, 'location': 'Poznan', 'country': 'Poland'},
    18: {'lat': 38.7073, 'long': -9.1363, 'location': 'Lisbon', 'country': 'Portugal'},
    19: {'lat': 59.3617, 'long': 17.8742, 'location': 'Bromma', 'country': 'Sweden'},
    20: {'lat': 46.0574, 'long': 14.5148, 'location': 'Ljubljana', 'country': 'Slovenia'},
    21: {'lat': 48.1531, 'long': 17.1297, 'location': 'Bratislava', 'country': 'Slovakia'},
    22: {'lat': 51.5086, 'long': -0.1264, 'location': 'London', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 3.0, 'fiber_length': 1205.741718697376},
    2.0: {'source': 1.0, 'destination': 5.0, 'fiber_length': 896.1689368748143},
    3.0: {'source': 1.0, 'destination': 10.0, 'fiber_length': 326.792609366545},
    4.0: {'source': 1.0, 'destination': 16.0, 'fiber_length': 8494.167120778287},
    5.0: {'source': 1.0, 'destination': 20.0, 'fiber_length': 416.2042271506733},
    6.0: {'source': 2.0, 'destination': 7.0, 'fiber_length': 395.5671921820349},
    7.0: {'source': 2.0, 'destination': 14.0, 'fiber_length': 280.095250485013},
    8.0: {'source': 2.0, 'destination': 15.0, 'fiber_length': 253.748818411639},
    9.0: {'source': 3.0, 'destination': 7.0, 'fiber_length': 614.5443962857513},
    10.0: {'source': 3.0, 'destination': 13.0, 'fiber_length': 375.2774972722834},
    11.0: {'source': 4.0, 'destination': 5.0, 'fiber_length': 615.985650719161},
    12.0: {'source': 4.0, 'destination': 17.0, 'fiber_length': 463.2422772840073},
    13.0: {'source': 4.0, 'destination': 21.0, 'fiber_length': 434.9231408588021},
    14.0: {'source': 5.0, 'destination': 7.0, 'fiber_length': 717.2335947676761},
    15.0: {'source': 5.0, 'destination': 8.0, 'fiber_length': 2240.980605010167},
    16.0: {'source': 5.0, 'destination': 11.0, 'fiber_length': 1500.0},
    17.0: {'source': 5.0, 'destination': 13.0, 'fiber_length': 777.315099448178},
    18.0: {'source': 5.0, 'destination': 15.0, 'fiber_length': 537.4556978209507},
    19.0: {'source': 5.0, 'destination': 19.0, 'fiber_length': 1500.0},
    20.0: {'source': 6.0, 'destination': 7.0, 'fiber_length': 1500.0},
    21.0: {'source': 6.0, 'destination': 13.0, 'fiber_length': 1500.0},
    22.0: {'source': 6.0, 'destination': 18.0, 'fiber_length': 754.3780747471078},
    23.0: {'source': 7.0, 'destination': 14.0, 'fiber_length': 430.2917638658975},
    24.0: {'source': 7.0, 'destination': 22.0, 'fiber_length': 515.36158679831},
    25.0: {'source': 8.0, 'destination': 13.0, 'fiber_length': 1815.420595779476},
    26.0: {'source': 9.0, 'destination': 10.0, 'fiber_length': 455.8481192958354},
    27.0: {'source': 9.0, 'destination': 20.0, 'fiber_length': 173.2635674625123},
    28.0: {'source': 10.0, 'destination': 21.0, 'fiber_length': 245.5911295353906},
    29.0: {'source': 11.0, 'destination': 22.0, 'fiber_length': 694.1747825761532},
    30.0: {'source': 12.0, 'destination': 13.0, 'fiber_length': 3319.590434051195},
    31.0: {'source': 12.0, 'destination': 15.0, 'fiber_length': 4116.059850695134},
    32.0: {'source': 15.0, 'destination': 22.0, 'fiber_length': 538.6083143873095},
    33.0: {'source': 16.0, 'destination': 22.0, 'fiber_length': 6961.479107534074},
    34.0: {'source': 17.0, 'destination': 19.0, 'fiber_length': 1165.404461338919},
    35.0: {'source': 18.0, 'destination': 22.0, 'fiber_length': 1983.225004686776},
    36.0: {'source': 19.0, 'destination': 22.0, 'fiber_length': 1781.016899491856},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
