def create_eon_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 38.7077507, 'long': -9.1365919, 'location': 'Lisbon', 'country': 'Portugal'},
    2: {'lat': 40.4167047, 'long': -3.7035825, 'location': 'Madrid', 'country': 'Spain'},
    3: {'lat': 48.8534951, 'long': 2.3483915, 'location': 'Paris', 'country': 'France'},
    4: {'lat': 51.5074456, 'long': -0.1277653, 'location': 'London', 'country': 'UK'},
    5: {'lat': 53.3493795, 'long': -6.2605593, 'location': 'Dublin', 'country': 'Ireland'},
    6: {'lat': 50.8465573, 'long': 4.351697, 'location': 'Brussels', 'country': 'Belgium'},
    7: {'lat': 45.4641943, 'long': 9.1896346, 'location': 'Milan', 'country': 'Italy'},
    8: {'lat': 52.3730796, 'long': 4.8924534, 'location': 'Amsterdam', 'country': 'Netherlands'},
    9: {'lat': 47.3744489, 'long': 8.5410422, 'location': 'Zurich', 'country': 'Switzerland'},
    10: {'lat': 50.0874654, 'long': 14.4212535, 'location': 'Prague', 'country': 'Czech Republic'},
    11: {'lat': 45.8130967, 'long': 15.9772795, 'location': 'Zagreb', 'country': 'Croatia'},
    12: {'lat': 48.2083537, 'long': 16.3725042, 'location': 'Vienna', 'country': 'Austria'},
    13: {'lat': 52.5170365, 'long': 13.3888599, 'location': 'Berlin', 'country': 'Germany'},
    14: {'lat': 55.6867243, 'long': 12.5700724, 'location': 'Copenhagen', 'country': 'Denmark'},
    15: {'lat': 59.9133301, 'long': 10.7389701, 'location': 'Oslo', 'country': 'Norway'},
    16: {'lat': 59.3251172, 'long': 18.0710935, 'location': 'Stockholm', 'country': 'Sweden'},
    17: {'lat': 55.7505412, 'long': 37.6174782, 'location': 'Moscow', 'country': 'Russia'},
    18: {'lat': 33.9597677, 'long': -83.376398, 'location': 'Athens', 'country': 'Greece'},
    19: {'lat': 41.8933203, 'long': 12.4829321, 'location': 'Rome', 'country': 'Italy'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 754.3499999999999},
    2.0: {'source': 2.0, 'destination': 3.0, 'fiber_length': 1500.0},
    3.0: {'source': 1.0, 'destination': 4.0, 'fiber_length': 1982.9875},
    4.0: {'source': 3.0, 'destination': 5.0, 'fiber_length': 1171.35},
    5.0: {'source': 3.0, 'destination': 4.0, 'fiber_length': 515.58},
    6.0: {'source': 4.0, 'destination': 5.0, 'fiber_length': 694.965},
    7.0: {'source': 4.0, 'destination': 6.0, 'fiber_length': 481.11},
    8.0: {'source': 4.0, 'destination': 8.0, 'fiber_length': 535.89},
    9.0: {'source': 4.0, 'destination': 15.0, 'fiber_length': 1500.0},
    10.0: {'source': 4.0, 'destination': 13.0, 'fiber_length': 1395.705},
    11.0: {'source': 3.0, 'destination': 9.0, 'fiber_length': 732.1500000000001},
    12.0: {'source': 3.0, 'destination': 7.0, 'fiber_length': 959.385},
    13.0: {'source': 3.0, 'destination': 6.0, 'fiber_length': 396.105},
    14.0: {'source': 6.0, 'destination': 8.0, 'fiber_length': 260.7},
    15.0: {'source': 6.0, 'destination': 9.0, 'fiber_length': 737.715},
    16.0: {'source': 9.0, 'destination': 7.0, 'fiber_length': 327.225},
    17.0: {'source': 9.0, 'destination': 10.0, 'fiber_length': 789.105},
    18.0: {'source': 7.0, 'destination': 11.0, 'fiber_length': 793.455},
    19.0: {'source': 7.0, 'destination': 19.0, 'fiber_length': 715.785},
    20.0: {'source': 8.0, 'destination': 13.0, 'fiber_length': 863.61},
    21.0: {'source': 8.0, 'destination': 14.0, 'fiber_length': 932.52},
    22.0: {'source': 8.0, 'destination': 10.0, 'fiber_length': 1064.745},
    23.0: {'source': 14.0, 'destination': 15.0, 'fiber_length': 723.4350000000001},
    24.0: {'source': 14.0, 'destination': 16.0, 'fiber_length': 781.29},
    25.0: {'source': 15.0, 'destination': 16.0, 'fiber_length': 625.89},
    26.0: {'source': 16.0, 'destination': 13.0, 'fiber_length': 1216.215},
    27.0: {'source': 16.0, 'destination': 17.0, 'fiber_length': 1533.9},
    28.0: {'source': 13.0, 'destination': 17.0, 'fiber_length': 2012.4375},
    29.0: {'source': 13.0, 'destination': 12.0, 'fiber_length': 785.4749999999999},
    30.0: {'source': 13.0, 'destination': 10.0, 'fiber_length': 419.28},
    31.0: {'source': 13.0, 'destination': 7.0, 'fiber_length': 1262.4},
    32.0: {'source': 10.0, 'destination': 12.0, 'fiber_length': 378.855},
    33.0: {'source': 12.0, 'destination': 11.0, 'fiber_length': 402.03},
    34.0: {'source': 18.0, 'destination': 11.0, 'fiber_length': 10027.7625},
    35.0: {'source': 18.0, 'destination': 19.0, 'fiber_length': 9999.5},
    36.0: {'source': 19.0, 'destination': 11.0, 'fiber_length': 777.045},
    37.0: {'source': 7.0, 'destination': 10.0, 'fiber_length': 968.25},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
