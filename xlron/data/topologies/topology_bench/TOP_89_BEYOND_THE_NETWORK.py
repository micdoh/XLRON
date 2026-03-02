def create_beyond_the_network_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 42.36, 'long': -71.06, 'location': 'Boston', 'country': 'United States of America (the)'},
    2: {'lat': 42.33, 'long': -83.05, 'location': 'Detroit', 'country': 'United States of America (the)'},
    3: {'lat': 39.95, 'long': -75.16, 'location': 'Philadelphia', 'country': 'United States of America (the)'},
    4: {'lat': 40.71, 'long': -74.01, 'location': 'New York City', 'country': 'United States of America (the)'},
    5: {'lat': 38.58, 'long': -121.49, 'location': 'Sacramento', 'country': 'United States of America (the)'},
    6: {'lat': 37.34, 'long': -121.89, 'location': 'San Jose', 'country': 'United States of America (the)'},
    7: {'lat': 47.61, 'long': -122.33, 'location': 'Seattle', 'country': 'United States of America (the)'},
    8: {'lat': 45.52, 'long': -122.68, 'location': 'Portland', 'country': 'United States of America (the)'},
    9: {'lat': 37.37, 'long': -122.04, 'location': 'Sunnyvale', 'country': 'United States of America (the)'},
    10: {'lat': 34.05, 'long': -118.24, 'location': 'Los Angeles', 'country': 'United States of America (the)'},
    11: {'lat': 40.44, 'long': -80.0, 'location': 'Pittsburgh', 'country': 'United States of America (the)'},
    12: {'lat': 32.78, 'long': -96.81, 'location': 'Dallas', 'country': 'United States of America (the)'},
    13: {'lat': 29.42, 'long': -98.49, 'location': 'San Antonio', 'country': 'United States of America (the)'},
    14: {'lat': 33.45, 'long': -112.07, 'location': 'Phoenix', 'country': 'United States of America (the)'},
    15: {'lat': 32.72, 'long': -117.16, 'location': 'San Diego', 'country': 'United States of America (the)'},
    16: {'lat': 27.95, 'long': -82.46, 'location': 'Tampa', 'country': 'United States of America (the)'},
    17: {'lat': 25.77, 'long': -80.19, 'location': 'Miami', 'country': 'United States of America (the)'},
    18: {'lat': 29.95, 'long': -90.08, 'location': 'New Orleans', 'country': 'United States of America (the)'},
    19: {'lat': 29.76, 'long': -95.36, 'location': 'Houston', 'country': 'United States of America (the)'},
    20: {'lat': 38.9, 'long': -77.04, 'location': 'Washington, D.C.', 'country': 'United States of America (the)'},
    21: {'lat': 39.29, 'long': -76.61, 'location': 'Baltimore', 'country': 'United States of America (the)'},
    22: {'lat': 39.74, 'long': -104.98, 'location': 'Denver', 'country': 'United States of America (the)'},
    23: {'lat': 44.98, 'long': -93.26, 'location': 'Minneapolis', 'country': 'United States of America (the)'},
    24: {'lat': 41.85, 'long': -87.65, 'location': 'Chicago', 'country': 'United States of America (the)'},
    25: {'lat': 39.11, 'long': -94.63, 'location': 'Kansas City', 'country': 'United States of America (the)'},
    26: {'lat': 33.75, 'long': -84.39, 'location': 'Atlanta', 'country': 'United States of America (the)'},
    27: {'lat': 35.77, 'long': -78.64, 'location': 'Raleigh', 'country': 'United States of America (the)'},
    28: {'lat': 39.04, 'long': -77.49, 'location': 'Ashburn', 'country': 'United States of America (the)'},
    29: {'lat': 38.9, 'long': -77.27, 'location': 'Vienna', 'country': 'United States of America (the)'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 1476.868954935013},
    2.0: {'source': 1.0, 'destination': 4.0, 'fiber_length': 459.728758316418},
    3.0: {'source': 2.0, 'destination': 24.0, 'fiber_length': 574.8944829350417},
    4.0: {'source': 3.0, 'destination': 11.0, 'fiber_length': 621.9499312218177},
    5.0: {'source': 3.0, 'destination': 4.0, 'fiber_length': 193.5165266446945},
    6.0: {'source': 3.0, 'destination': 21.0, 'fiber_length': 216.3834396565364},
    7.0: {'source': 3.0, 'destination': 28.0, 'fiber_length': 336.1009376453503},
    8.0: {'source': 4.0, 'destination': 24.0, 'fiber_length': 1500.0},
    9.0: {'source': 5.0, 'destination': 6.0, 'fiber_length': 213.4063487535815},
    10.0: {'source': 5.0, 'destination': 8.0, 'fiber_length': 1166.845178904244},
    11.0: {'source': 6.0, 'destination': 9.0, 'fiber_length': 20.50710500213939},
    12.0: {'source': 6.0, 'destination': 22.0, 'fiber_length': 1865.480902874388},
    13.0: {'source': 7.0, 'destination': 24.0, 'fiber_length': 3485.643998403856},
    14.0: {'source': 7.0, 'destination': 8.0, 'fiber_length': 350.8979570264834},
    15.0: {'source': 9.0, 'destination': 10.0, 'fiber_length': 755.8274651640023},
    16.0: {'source': 10.0, 'destination': 12.0, 'fiber_length': 2487.99326220719},
    17.0: {'source': 10.0, 'destination': 14.0, 'fiber_length': 861.3692628497527},
    18.0: {'source': 10.0, 'destination': 15.0, 'fiber_length': 268.0139459776118},
    19.0: {'source': 11.0, 'destination': 20.0, 'fiber_length': 458.6437394496578},
    20.0: {'source': 11.0, 'destination': 24.0, 'fiber_length': 988.8431894766425},
    21.0: {'source': 12.0, 'destination': 13.0, 'fiber_length': 609.5955997354126},
    22.0: {'source': 12.0, 'destination': 19.0, 'fiber_length': 544.4600499700148},
    23.0: {'source': 12.0, 'destination': 25.0, 'fiber_length': 1095.95956804919},
    24.0: {'source': 12.0, 'destination': 26.0, 'fiber_length': 1500.0},
    25.0: {'source': 12.0, 'destination': 18.0, 'fiber_length': 1068.08484803266},
    26.0: {'source': 13.0, 'destination': 19.0, 'fiber_length': 457.4870157403198},
    27.0: {'source': 13.0, 'destination': 14.0, 'fiber_length': 1703.497385547838},
    28.0: {'source': 14.0, 'destination': 15.0, 'fiber_length': 721.5882097187069},
    29.0: {'source': 16.0, 'destination': 17.0, 'fiber_length': 496.2580235875224},
    30.0: {'source': 16.0, 'destination': 18.0, 'fiber_length': 1160.805957131225},
    31.0: {'source': 17.0, 'destination': 26.0, 'fiber_length': 1462.933348688248},
    32.0: {'source': 18.0, 'destination': 19.0, 'fiber_length': 764.3786605203708},
    33.0: {'source': 20.0, 'destination': 27.0, 'fiber_length': 563.5055664441111},
    34.0: {'source': 20.0, 'destination': 21.0, 'fiber_length': 85.61340860642284},
    35.0: {'source': 20.0, 'destination': 29.0, 'fiber_length': 29.85515008040191},
    36.0: {'source': 22.0, 'destination': 24.0, 'fiber_length': 1843.716434385458},
    37.0: {'source': 23.0, 'destination': 24.0, 'fiber_length': 856.6967858075002},
    38.0: {'source': 24.0, 'destination': 25.0, 'fiber_length': 996.0670010192133},
    39.0: {'source': 26.0, 'destination': 27.0, 'fiber_length': 856.7329203000172},
    40.0: {'source': 26.0, 'destination': 28.0, 'fiber_length': 1278.565590677604},
    41.0: {'source': 28.0, 'destination': 29.0, 'fiber_length': 36.86685026301232},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
