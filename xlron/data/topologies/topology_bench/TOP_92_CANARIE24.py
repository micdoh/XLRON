def create_canarie24_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 48.4, 'long': -89.32, 'location': 'Thunder Bay', 'country': 'Canada'},
    2: {'lat': 41.85, 'long': -87.65, 'location': 'Chicago', 'country': 'United States of America (the)'},
    3: {'lat': 50.45, 'long': -104.62, 'location': 'Regina', 'country': 'Canada'},
    4: {'lat': 49.88, 'long': -97.15, 'location': 'Winnipeg', 'country': 'Canada'},
    5: {'lat': 45.41, 'long': -75.7, 'location': 'Ottawa', 'country': 'Canada'},
    6: {'lat': 45.51, 'long': -73.59, 'location': 'Montreal', 'country': 'Canada'},
    7: {'lat': 42.33, 'long': -83.05, 'location': 'Detroit', 'country': 'United States of America (the)'},
    8: {'lat': 43.7, 'long': -79.42, 'location': 'Toronto', 'country': 'Canada'},
    9: {'lat': 45.95, 'long': -66.67, 'location': 'Fredericton', 'country': 'Canada'},
    10: {'lat': 46.24, 'long': -63.13, 'location': 'Charlottetown', 'country': 'Canada'},
    11: {'lat': 42.36, 'long': -71.06, 'location': 'Boston', 'country': 'United States of America (the)'},
    12: {'lat': 40.71, 'long': -74.01, 'location': 'New York City', 'country': 'United States of America (the)'},
    13: {'lat': 47.56, 'long': -52.71, 'location': "St. John's", 'country': 'Canada'},
    14: {'lat': 44.65, 'long': -63.57, 'location': 'Halifax', 'country': 'Canada'},
    15: {'lat': 51.05, 'long': -114.09, 'location': 'Calgary', 'country': 'Canada'},
    16: {'lat': 52.12, 'long': -106.63, 'location': 'Saskatoon', 'country': 'Canada'},
    17: {'lat': 60.72, 'long': -135.05, 'location': 'Whitehorse', 'country': 'Canada'},
    18: {'lat': 62.46, 'long': -114.35, 'location': 'Yellowknife', 'country': 'Canada'},
    19: {'lat': 53.55, 'long': -113.47, 'location': 'Edmonton', 'country': 'Canada'},
    20: {'lat': 50.67, 'long': -120.32, 'location': 'Kamloops', 'country': 'Canada'},
    21: {'lat': 49.88, 'long': -119.49, 'location': 'Kelowna', 'country': 'Canada'},
    22: {'lat': 49.25, 'long': -123.12, 'location': 'Vancouver', 'country': 'Canada'},
    23: {'lat': 48.43, 'long': -123.37, 'location': 'Victoria', 'country': 'Canada'},
    24: {'lat': 47.61, 'long': -122.33, 'location': 'Seattle', 'country': 'United States of America (the)'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 4.0, 'fiber_length': 888.857949926725},
    2.0: {'source': 1.0, 'destination': 8.0, 'fiber_length': 1386.750011760354},
    3.0: {'source': 2.0, 'destination': 4.0, 'fiber_length': 1500.0},
    4.0: {'source': 2.0, 'destination': 7.0, 'fiber_length': 574.8944829350417},
    5.0: {'source': 2.0, 'destination': 15.0, 'fiber_length': 2815.444226988842},
    6.0: {'source': 2.0, 'destination': 24.0, 'fiber_length': 3485.643998403856},
    7.0: {'source': 3.0, 'destination': 4.0, 'fiber_length': 803.4162900959275},
    8.0: {'source': 3.0, 'destination': 15.0, 'fiber_length': 1003.66705332373},
    9.0: {'source': 4.0, 'destination': 8.0, 'fiber_length': 1886.93657021348},
    10.0: {'source': 4.0, 'destination': 15.0, 'fiber_length': 1504.208516300457},
    11.0: {'source': 4.0, 'destination': 16.0, 'fiber_length': 1061.930861921065},
    12.0: {'source': 5.0, 'destination': 6.0, 'fiber_length': 247.4032001412376},
    13.0: {'source': 5.0, 'destination': 8.0, 'fiber_length': 526.0580396293944},
    14.0: {'source': 6.0, 'destination': 9.0, 'fiber_length': 808.7577697097216},
    15.0: {'source': 6.0, 'destination': 14.0, 'fiber_length': 1187.999858185761},
    16.0: {'source': 6.0, 'destination': 11.0, 'fiber_length': 606.8693858795343},
    17.0: {'source': 7.0, 'destination': 8.0, 'fiber_length': 498.1255973955056},
    18.0: {'source': 8.0, 'destination': 12.0, 'fiber_length': 833.6552487670347},
    19.0: {'source': 9.0, 'destination': 10.0, 'fiber_length': 412.2645211122608},
    20.0: {'source': 10.0, 'destination': 14.0, 'fiber_length': 270.1507019229736},
    21.0: {'source': 11.0, 'destination': 12.0, 'fiber_length': 459.728758316418},
    22.0: {'source': 12.0, 'destination': 14.0, 'fiber_length': 1437.431796207301},
    23.0: {'source': 13.0, 'destination': 14.0, 'fiber_length': 1344.9712926912},
    24.0: {'source': 15.0, 'destination': 20.0, 'fiber_length': 658.763518588547},
    25.0: {'source': 15.0, 'destination': 19.0, 'fiber_length': 421.7445112294374},
    26.0: {'source': 16.0, 'destination': 19.0, 'fiber_length': 728.9609513814639},
    27.0: {'source': 17.0, 'destination': 19.0, 'fiber_length': 1895.334694359173},
    28.0: {'source': 18.0, 'destination': 19.0, 'fiber_length': 1488.117986312792},
    29.0: {'source': 20.0, 'destination': 21.0, 'fiber_length': 158.7121138691045},
    30.0: {'source': 20.0, 'destination': 22.0, 'fiber_length': 382.5273472837764},
    31.0: {'source': 22.0, 'destination': 23.0, 'fiber_length': 139.4958300540092},
    32.0: {'source': 22.0, 'destination': 24.0, 'fiber_length': 287.1681018437656},
    33.0: {'source': 23.0, 'destination': 24.0, 'fiber_length': 179.3502469719835},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
