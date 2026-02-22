def create_grnet_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 39.37, 'long': 21.92, 'location': 'Karditsa', 'country': 'Greece'},
    2: {'lat': 39.37, 'long': 22.95, 'location': 'Volos', 'country': 'Greece'},
    3: {'lat': 39.64, 'long': 22.42, 'location': 'Larisa', 'country': 'Greece'},
    4: {'lat': 39.56, 'long': 21.77, 'location': 'Trikala', 'country': 'Greece'},
    5: {'lat': 39.11, 'long': 26.55, 'location': 'Mytilini', 'country': 'Greece'},
    6: {'lat': 38.37, 'long': 26.14, 'location': 'Chios', 'country': 'Greece'},
    7: {'lat': 39.5, 'long': 20.27, 'location': 'Igoumenitsa', 'country': 'Greece'},
    8: {'lat': 39.62, 'long': 19.92, 'location': 'Kerkyra', 'country': 'Greece'},
    9: {'lat': 37.44, 'long': 24.94, 'location': 'Ermoupolis', 'country': 'Greece'},
    10: {'lat': 37.45, 'long': 25.33, 'location': 'Mykonos', 'country': 'Greece'},
    11: {'lat': 38.37, 'long': 21.43, 'location': 'Mesolongi', 'country': 'Greece'},
    12: {'lat': 35.51, 'long': 24.02, 'location': 'Chania', 'country': 'Greece'},
    13: {'lat': 35.33, 'long': 25.14, 'location': 'Irakleion', 'country': 'Greece'},
    14: {'lat': 35.36, 'long': 24.47, 'location': 'Rethymno', 'country': 'Greece'},
    15: {'lat': 36.91, 'long': 21.7, 'location': 'Pylos', 'country': 'Greece'},
    16: {'lat': 37.04, 'long': 22.11, 'location': 'Kalamata', 'country': 'Greece'},
    17: {'lat': 38.46, 'long': 23.6, 'location': 'Chalkida', 'country': 'Greece'},
    18: {'lat': 37.98, 'long': 23.72, 'location': 'Athens', 'country': 'Greece'},
    19: {'lat': 36.44, 'long': 28.22, 'location': 'Rodos', 'country': 'Greece'},
    20: {'lat': 37.79, 'long': 26.7, 'location': 'Neon Karlovasion', 'country': 'Greece'},
    21: {'lat': 37.51, 'long': 22.38, 'location': 'Tripoli', 'country': 'Greece'},
    22: {'lat': 38.24, 'long': 21.73, 'location': 'Patra', 'country': 'Greece'},
    23: {'lat': 38.03, 'long': 22.11, 'location': 'Kalavryta', 'country': 'Greece'},
    24: {'lat': 37.92, 'long': 22.88, 'location': 'Arkhaia Korinthos', 'country': 'Greece'},
    25: {'lat': 38.47, 'long': 25.91, 'location': 'Vrontados', 'country': 'Greece'},
    26: {'lat': 39.67, 'long': 20.85, 'location': 'Ioannina', 'country': 'Greece'},
    27: {'lat': 41.09, 'long': 23.55, 'location': 'Serres', 'country': 'Greece'},
    28: {'lat': 40.64, 'long': 22.94, 'location': 'Thessaloniki', 'country': 'Greece'},
    29: {'lat': 40.8, 'long': 22.05, 'location': 'Edessa', 'country': 'Greece'},
    30: {'lat': 40.52, 'long': 21.27, 'location': 'Kastoria', 'country': 'Greece'},
    31: {'lat': 40.94, 'long': 24.4, 'location': 'Kavala', 'country': 'Greece'},
    32: {'lat': 41.14, 'long': 24.88, 'location': 'Xanthi', 'country': 'Greece'},
    33: {'lat': 41.12, 'long': 25.4, 'location': 'Komotini', 'country': 'Greece'},
    34: {'lat': 40.85, 'long': 25.87, 'location': 'Alexandroupoli', 'country': 'Greece'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 3.0, 'fiber_length': 78.53929172182724},
    2.0: {'source': 2.0, 'destination': 3.0, 'fiber_length': 81.73229393655609},
    3.0: {'source': 3.0, 'destination': 4.0, 'fiber_length': 84.59401638652376},
    4.0: {'source': 3.0, 'destination': 18.0, 'fiber_length': 324.3472995358508},
    5.0: {'source': 3.0, 'destination': 28.0, 'fiber_length': 179.4869997937854},
    6.0: {'source': 5.0, 'destination': 25.0, 'fiber_length': 135.3424406030754},
    7.0: {'source': 5.0, 'destination': 6.0, 'fiber_length': 134.4584677324705},
    8.0: {'source': 6.0, 'destination': 9.0, 'fiber_length': 221.3588042185257},
    9.0: {'source': 7.0, 'destination': 26.0, 'fiber_length': 79.76494490241927},
    10.0: {'source': 7.0, 'destination': 8.0, 'fiber_length': 49.25628855799397},
    11.0: {'source': 9.0, 'destination': 10.0, 'fiber_length': 51.6717453698879},
    12.0: {'source': 9.0, 'destination': 32.0, 'fiber_length': 617.1803943232806},
    13.0: {'source': 9.0, 'destination': 18.0, 'fiber_length': 184.4625948166933},
    14.0: {'source': 9.0, 'destination': 13.0, 'fiber_length': 352.954785110536},
    15.0: {'source': 10.0, 'destination': 20.0, 'fiber_length': 189.6678890347853},
    16.0: {'source': 11.0, 'destination': 22.0, 'fiber_length': 44.85471876101077},
    17.0: {'source': 12.0, 'destination': 18.0, 'fiber_length': 413.9229400580977},
    18.0: {'source': 12.0, 'destination': 13.0, 'fiber_length': 155.1654250427453},
    19.0: {'source': 13.0, 'destination': 14.0, 'fiber_length': 91.2904242025082},
    20.0: {'source': 13.0, 'destination': 18.0, 'fiber_length': 481.0915037522016},
    21.0: {'source': 13.0, 'destination': 19.0, 'fiber_length': 455.5023571894742},
    22.0: {'source': 15.0, 'destination': 16.0, 'fiber_length': 58.77805641493629},
    23.0: {'source': 16.0, 'destination': 22.0, 'fiber_length': 206.346951649401},
    24.0: {'source': 17.0, 'destination': 18.0, 'fiber_length': 81.5899524590216},
    25.0: {'source': 18.0, 'destination': 22.0, 'fiber_length': 264.731685058454},
    26.0: {'source': 18.0, 'destination': 24.0, 'fiber_length': 110.9318983172991},
    27.0: {'source': 18.0, 'destination': 28.0, 'fiber_length': 454.9374876550069},
    28.0: {'source': 19.0, 'destination': 20.0, 'fiber_length': 302.6001938517479},
    29.0: {'source': 21.0, 'destination': 24.0, 'fiber_length': 95.01942907971164},
    30.0: {'source': 22.0, 'destination': 23.0, 'fiber_length': 60.92742003066908},
    31.0: {'source': 22.0, 'destination': 26.0, 'fiber_length': 264.4131623925118},
    32.0: {'source': 26.0, 'destination': 28.0, 'fiber_length': 311.6945352214178},
    33.0: {'source': 27.0, 'destination': 28.0, 'fiber_length': 107.4881727901519},
    34.0: {'source': 28.0, 'destination': 32.0, 'fiber_length': 258.4322462802127},
    35.0: {'source': 28.0, 'destination': 29.0, 'fiber_length': 115.6287838232727},
    36.0: {'source': 29.0, 'destination': 30.0, 'fiber_length': 109.1826654097571},
    37.0: {'source': 31.0, 'destination': 32.0, 'fiber_length': 68.98688815502658},
    38.0: {'source': 32.0, 'destination': 33.0, 'fiber_length': 65.41324576667797},
    39.0: {'source': 33.0, 'destination': 34.0, 'fiber_length': 74.36357320098753},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
