def create_renater_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 50.6365654, 'long': 3.0635282, 'location': 'Lille', 'country': 'France'},
    2: {'lat': 49.4179497, 'long': 2.8263171, 'location': 'Compiègne', 'country': 'France'},
    3: {'lat': 48.8534951, 'long': 2.3483915, 'location': 'Paris', 'country': 'France'},
    4: {'lat': 47.3497873, 'long': 2.1963339, 'location': 'Nançay', 'country': 'France'},
    5: {'lat': 48.699184, 'long': 2.187457, 'location': 'Orsay', 'country': 'France'},
    6: {'lat': 47.9027336, 'long': 1.9086066, 'location': 'Orléans', 'country': 'France'},
    7: {'lat': 47.3900474, 'long': 0.6889268, 'location': 'Tours', 'country': 'France'},
    8: {'lat': 48.1113387, 'long': -1.6800198, 'location': 'Rennes', 'country': 'France'},
    9: {'lat': 46.5802596, 'long': 0.340196, 'location': 'Poitiers', 'country': 'France'},
    10: {'lat': 49.4404591, 'long': 1.0939658, 'location': 'Rouen', 'country': 'France'},
    11: {'lat': 49.1813403, 'long': -0.3635615, 'location': 'Caen', 'country': 'France'},
    12: {'lat': 47.2186371, 'long': -1.5541362, 'location': 'Nantes', 'country': 'France'},
    13: {'lat': 44.841225, 'long': -0.5800364, 'location': 'Bordeaux', 'country': 'France'},
    14: {'lat': 45.8354243, 'long': 1.2644847, 'location': 'Limoges', 'country': 'France'},
    15: {'lat': 43.2957547, 'long': -0.3685668, 'location': 'Pau', 'country': 'France'},
    16: {'lat': 43.6037406, 'long': 1.4445016, 'location': 'Toulouse', 'country': 'France'},
    17: {'lat': 43.2961743, 'long': 5.3699525, 'location': 'Marseille', 'country': 'France'},
    18: {'lat': 45.7578137, 'long': 4.8320114, 'location': 'Lyon', 'country': 'France'},
    19: {'lat': 45.7774551, 'long': 3.0819427, 'location': 'Clermont-Ferrand', 'country': 'France'},
    20: {'lat': 47.3215806, 'long': 5.0414701, 'location': 'Dijon', 'country': 'France'},
    21: {'lat': 47.2380222, 'long': 6.0243622, 'location': 'Besançon', 'country': 'France'},
    22: {'lat': 48.584614, 'long': 7.7507127, 'location': 'Strasbourg', 'country': 'France'},
    23: {'lat': 48.6937223, 'long': 6.1834097, 'location': 'Nancy', 'country': 'France'},
    24: {'lat': 49.2577886, 'long': 4.031926, 'location': 'Reims', 'country': 'France'},
    25: {'lat': 48.8909198, 'long': 7.998767, 'location': 'Esch', 'country': 'France'},
    26: {'lat': 46.3289084, 'long': -0.4729053, 'location': 'Genève', 'country': 'France'},
    27: {'lat': 45.1875602, 'long': 5.7357819, 'location': 'Grenoble', 'country': 'France'},
    28: {'lat': 43.7089124, 'long': 5.7398568, 'location': 'Cadarache', 'country': 'France'},
    29: {'lat': 43.6195225, 'long': 7.0518158, 'location': 'Sophia Antipolis', 'country': 'France'},
    30: {'lat': 43.7009358, 'long': 7.2683912, 'location': 'Nice', 'country': 'France'},
    31: {'lat': 42.3052904, 'long': 9.1511935, 'location': 'Corte', 'country': 'France'},
    32: {'lat': 43.1257311, 'long': 5.9304919, 'location': 'Toulon', 'country': 'France'},
    33: {'lat': 43.9492493, 'long': 4.8059012, 'location': 'Avignon', 'country': 'France'},
    34: {'lat': 43.6112422, 'long': 3.8767337, 'location': 'Montpellier', 'country': 'France'},
    }

    edge_attributes = {
    1: {'source': 3, 'destination': 2, 'fiber_length': 72},
    2: {'source': 2, 'destination': 1, 'fiber_length': 137},
    3: {'source': 1, 'destination': 3, 'fiber_length': 205},
    4: {'source': 3, 'destination': 10, 'fiber_length': 112},
    5: {'source': 10, 'destination': 11, 'fiber_length': 110},
    6: {'source': 11, 'destination': 8, 'fiber_length': 153},
    7: {'source': 3, 'destination': 8, 'fiber_length': 308},
    8: {'source': 3, 'destination': 5, 'fiber_length': 21},
    9: {'source': 5, 'destination': 6, 'fiber_length': 91},
    10: {'source': 5, 'destination': 4, 'fiber_length': 150},
    11: {'source': 6, 'destination': 7, 'fiber_length': 108},
    12: {'source': 7, 'destination': 8, 'fiber_length': 194},
    13: {'source': 8, 'destination': 12, 'fiber_length': 100},
    14: {'source': 7, 'destination': 9, 'fiber_length': 94},
    15: {'source': 12, 'destination': 13, 'fiber_length': 275},
    16: {'source': 9, 'destination': 14, 'fiber_length': 109},
    17: {'source': 9, 'destination': 13, 'fiber_length': 206},
    18: {'source': 13, 'destination': 16, 'fiber_length': 212},
    19: {'source': 13, 'destination': 15, 'fiber_length': 173},
    20: {'source': 15, 'destination': 16, 'fiber_length': 150},
    21: {'source': 16, 'destination': 34, 'fiber_length': 196},
    22: {'source': 34, 'destination': 17, 'fiber_length': 126},
    23: {'source': 17, 'destination': 18, 'fiber_length': 277},
    24: {'source': 17, 'destination': 33, 'fiber_length': 86},
    25: {'source': 33, 'destination': 28, 'fiber_length': 80},
    26: {'source': 17, 'destination': 28, 'fiber_length': 55},
    27: {'source': 17, 'destination': 29, 'fiber_length': 140},
    28: {'source': 17, 'destination': 32, 'fiber_length': 49},
    29: {'source': 17, 'destination': 31, 'fiber_length': 328},
    30: {'source': 31, 'destination': 30, 'fiber_length': 218},
    31: {'source': 30, 'destination': 28, 'fiber_length': 123},
    32: {'source': 29, 'destination': 30, 'fiber_length': 20},
    33: {'source': 28, 'destination': 27, 'fiber_length': 164},
    34: {'source': 27, 'destination': 26, 'fiber_length': 498},
    35: {'source': 26, 'destination': 18, 'fiber_length': 414},
    36: {'source': 18, 'destination': 3, 'fiber_length': 392},
    37: {'source': 18, 'destination': 20, 'fiber_length': 175},
    38: {'source': 20, 'destination': 21, 'fiber_length': 75},
    39: {'source': 21, 'destination': 22, 'fiber_length': 197},
    40: {'source': 22, 'destination': 23, 'fiber_length': 116},
    41: {'source': 23, 'destination': 25, 'fiber_length': 135},
    42: {'source': 23, 'destination': 24, 'fiber_length': 169},
    43: {'source': 3, 'destination': 24, 'fiber_length': 131},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
