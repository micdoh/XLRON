def create_biznet_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': -7.73, 'long': 109.01, 'location': 'Karangbadar Kidul', 'country': 'Indonesia'},
    2: {'lat': -7.42, 'long': 109.23, 'location': 'Purwokerto', 'country': 'Indonesia'},
    3: {'lat': -6.92, 'long': 106.93, 'location': 'Sukabumi', 'country': 'Indonesia'},
    4: {'lat': -7.33, 'long': 108.2, 'location': 'Cilingga', 'country': 'Indonesia'},
    5: {'lat': -6.89, 'long': 109.68, 'location': 'Pekalongan', 'country': 'Indonesia'},
    6: {'lat': -6.99, 'long': 110.42, 'location': 'Semarang', 'country': 'Indonesia'},
    7: {'lat': -7.47, 'long': 110.22, 'location': 'Magelang', 'country': 'Indonesia'},
    8: {'lat': -7.78, 'long': 110.36, 'location': 'Yogyakarta', 'country': 'Indonesia'},
    9: {'lat': -6.8, 'long': 110.84, 'location': 'Kudus', 'country': 'Indonesia'},
    10: {'lat': -7.56, 'long': 110.83, 'location': 'Surakarta', 'country': 'Indonesia'},
    11: {'lat': -9.6, 'long': 123.77, 'location': 'Naisano Dua', 'country': 'Indonesia'},
    12: {'lat': -7.25, 'long': 112.75, 'location': 'Surabaya', 'country': 'Indonesia'},
    13: {'lat': -7.82, 'long': 112.02, 'location': 'Kediri', 'country': 'Indonesia'},
    14: {'lat': -7.63, 'long': 111.52, 'location': 'Madiun', 'country': 'Indonesia'},
    15: {'lat': -6.9, 'long': 112.05, 'location': 'Merik', 'country': 'Indonesia'},
    16: {'lat': -8.22, 'long': 114.36, 'location': 'Banyuwangi', 'country': 'Indonesia'},
    17: {'lat': -8.17, 'long': 113.7, 'location': 'Jember', 'country': 'Indonesia'},
    18: {'lat': -7.75, 'long': 113.22, 'location': 'Probolinggo', 'country': 'Indonesia'},
    19: {'lat': -7.98, 'long': 112.63, 'location': 'Malang', 'country': 'Indonesia'},
    20: {'lat': -6.77, 'long': 105.85, 'location': 'Umbulan', 'country': 'Indonesia'},
    21: {'lat': -6.87, 'long': 109.14, 'location': 'Tegal', 'country': 'Indonesia'},
    22: {'lat': -6.71, 'long': 108.56, 'location': 'Cirebon', 'country': 'Indonesia'},
    23: {'lat': -6.9, 'long': 107.62, 'location': 'Bandung', 'country': 'Indonesia'},
    24: {'lat': -6.3, 'long': 107.31, 'location': 'Rengasdengklok', 'country': 'Indonesia'},
    25: {'lat': -6.21, 'long': 106.85, 'location': 'Jakarta', 'country': 'Indonesia'},
    26: {'lat': -6.59, 'long': 106.79, 'location': 'Bogor', 'country': 'Indonesia'},
    27: {'lat': -6.11, 'long': 106.15, 'location': 'Serang', 'country': 'Indonesia'},
    28: {'lat': -6.07, 'long': 105.88, 'location': 'Ciparay', 'country': 'Indonesia'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 63.21823334353002},
    2.0: {'source': 1.0, 'destination': 4.0, 'fiber_length': 149.6333136910914},
    3.0: {'source': 2.0, 'destination': 7.0, 'fiber_length': 163.9446271563576},
    4.0: {'source': 3.0, 'destination': 26.0, 'fiber_length': 59.72676188400029},
    5.0: {'source': 3.0, 'destination': 23.0, 'fiber_length': 114.2994786309474},
    6.0: {'source': 4.0, 'destination': 23.0, 'fiber_length': 119.8281628854474},
    7.0: {'source': 5.0, 'destination': 21.0, 'fiber_length': 89.48152741734025},
    8.0: {'source': 5.0, 'destination': 6.0, 'fiber_length': 123.6521053737185},
    9.0: {'source': 6.0, 'destination': 9.0, 'fiber_length': 76.42614712443529},
    10.0: {'source': 6.0, 'destination': 10.0, 'fiber_length': 116.7907626137179},
    11.0: {'source': 7.0, 'destination': 8.0, 'fiber_length': 56.64925352090003},
    12.0: {'source': 8.0, 'destination': 10.0, 'fiber_length': 85.92069867134964},
    13.0: {'source': 9.0, 'destination': 15.0, 'fiber_length': 201.071071999294},
    14.0: {'source': 10.0, 'destination': 14.0, 'fiber_length': 114.6730015583044},
    15.0: {'source': 11.0, 'destination': 16.0, 'fiber_length': 1500.0},
    16.0: {'source': 12.0, 'destination': 18.0, 'fiber_length': 113.9980579802777},
    17.0: {'source': 12.0, 'destination': 13.0, 'fiber_length': 153.6511759437691},
    18.0: {'source': 12.0, 'destination': 15.0, 'fiber_length': 129.7409812648952},
    19.0: {'source': 13.0, 'destination': 19.0, 'fiber_length': 104.2512936616901},
    20.0: {'source': 13.0, 'destination': 14.0, 'fiber_length': 88.50731681378437},
    21.0: {'source': 16.0, 'destination': 17.0, 'fiber_length': 109.2775559320207},
    22.0: {'source': 17.0, 'destination': 18.0, 'fiber_length': 105.8021963542024},
    23.0: {'source': 17.0, 'destination': 19.0, 'fiber_length': 179.51755359469},
    24.0: {'source': 20.0, 'destination': 26.0, 'fiber_length': 158.5881433532389},
    25.0: {'source': 20.0, 'destination': 28.0, 'fiber_length': 116.8605067280724},
    26.0: {'source': 21.0, 'destination': 22.0, 'fiber_length': 99.6990781272909},
    27.0: {'source': 22.0, 'destination': 23.0, 'fiber_length': 158.872980020918},
    28.0: {'source': 23.0, 'destination': 24.0, 'fiber_length': 112.4865439733617},
    29.0: {'source': 24.0, 'destination': 25.0, 'fiber_length': 77.7309912823988},
    30.0: {'source': 25.0, 'destination': 26.0, 'fiber_length': 64.15661318995707},
    31.0: {'source': 25.0, 'destination': 27.0, 'fiber_length': 117.272693215779},
    32.0: {'source': 27.0, 'destination': 28.0, 'fiber_length': 45.27407036269294},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
