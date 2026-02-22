def create_pionier27_l3_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 52.23, 'long': 21.01, 'location': 'Warsaw', 'country': 'Poland'},
    2: {'lat': 51.4, 'long': 21.15, 'location': 'Radom', 'country': 'Poland'},
    3: {'lat': 53.13, 'long': 23.15, 'location': 'Bialystok', 'country': 'Poland'},
    4: {'lat': 51.25, 'long': 22.57, 'location': 'Lublin', 'country': 'Poland'},
    5: {'lat': 51.75, 'long': 19.47, 'location': 'Lodz', 'country': 'Poland'},
    6: {'lat': 51.42, 'long': 21.97, 'location': 'Pulawy', 'country': 'Poland'},
    7: {'lat': 50.04, 'long': 22.0, 'location': 'Rzeszow', 'country': 'Poland'},
    8: {'lat': 50.87, 'long': 20.63, 'location': 'Kielce', 'country': 'Poland'},
    9: {'lat': 54.15, 'long': 19.41, 'location': 'Elblag', 'country': 'Poland'},
    10: {'lat': 54.11, 'long': 22.93, 'location': 'Suwalki', 'country': 'Poland'},
    11: {'lat': 52.23, 'long': 20.24, 'location': 'Sochaczew', 'country': 'Poland'},
    12: {'lat': 52.74, 'long': 15.23, 'location': 'Gorzow Wielkopolski', 'country': 'Poland'},
    13: {'lat': 49.82, 'long': 19.05, 'location': 'Bielsko-Biala', 'country': 'Poland'},
    14: {'lat': 50.72, 'long': 23.25, 'location': 'Zamosc', 'country': 'Poland'},
    15: {'lat': 50.8, 'long': 19.12, 'location': 'Czestochowa', 'country': 'Poland'},
    16: {'lat': 50.08, 'long': 19.92, 'location': 'Krakow', 'country': 'Poland'},
    17: {'lat': 51.1, 'long': 17.03, 'location': 'Wroclaw', 'country': 'Poland'},
    18: {'lat': 50.67, 'long': 17.95, 'location': 'Opole', 'country': 'Poland'},
    19: {'lat': 50.28, 'long': 18.67, 'location': 'Gliwice', 'country': 'Poland'},
    20: {'lat': 53.78, 'long': 20.48, 'location': 'Olsztyn', 'country': 'Poland'},
    21: {'lat': 54.35, 'long': 18.65, 'location': 'Gdansk', 'country': 'Poland'},
    22: {'lat': 54.19, 'long': 16.17, 'location': 'Koszalin', 'country': 'Poland'},
    23: {'lat': 53.43, 'long': 14.55, 'location': 'Szczecin', 'country': 'Poland'},
    24: {'lat': 51.94, 'long': 15.51, 'location': 'Zielona Gora', 'country': 'Poland'},
    25: {'lat': 52.42, 'long': 16.97, 'location': 'Poznan', 'country': 'Poland'},
    26: {'lat': 53.12, 'long': 18.01, 'location': 'Bydgoszcz', 'country': 'Poland'},
    27: {'lat': 53.01, 'long': 18.6, 'location': 'Torun', 'country': 'Poland'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 139.1882096147356},
    2.0: {'source': 1.0, 'destination': 25.0, 'fiber_length': 413.0025180279077},
    3.0: {'source': 1.0, 'destination': 3.0, 'fiber_length': 263.3489639066166},
    4.0: {'source': 1.0, 'destination': 5.0, 'fiber_length': 177.2765341606382},
    5.0: {'source': 2.0, 'destination': 8.0, 'fiber_length': 103.8090366128347},
    6.0: {'source': 2.0, 'destination': 6.0, 'fiber_length': 85.3739547448099},
    7.0: {'source': 3.0, 'destination': 10.0, 'fiber_length': 164.8989742731337},
    8.0: {'source': 4.0, 'destination': 14.0, 'fiber_length': 113.6321014438478},
    9.0: {'source': 4.0, 'destination': 6.0, 'fiber_length': 68.65249769807909},
    10.0: {'source': 5.0, 'destination': 11.0, 'fiber_length': 112.5351594181881},
    11.0: {'source': 5.0, 'destination': 25.0, 'fiber_length': 279.5207502149059},
    12.0: {'source': 5.0, 'destination': 15.0, 'fiber_length': 162.6062920330432},
    13.0: {'source': 7.0, 'destination': 14.0, 'fiber_length': 174.7534644807546},
    14.0: {'source': 7.0, 'destination': 16.0, 'fiber_length': 222.8153948007431},
    15.0: {'source': 8.0, 'destination': 16.0, 'fiber_length': 151.7951398283662},
    16.0: {'source': 9.0, 'destination': 20.0, 'fiber_length': 121.7814953135299},
    17.0: {'source': 9.0, 'destination': 21.0, 'fiber_length': 81.22620749412314},
    18.0: {'source': 10.0, 'destination': 20.0, 'fiber_length': 246.7145911780555},
    19.0: {'source': 11.0, 'destination': 25.0, 'fiber_length': 334.8191775924115},
    20.0: {'source': 12.0, 'destination': 23.0, 'fiber_length': 133.7358325604599},
    21.0: {'source': 12.0, 'destination': 25.0, 'fiber_length': 184.2467880845705},
    22.0: {'source': 13.0, 'destination': 19.0, 'fiber_length': 86.85011346655239},
    23.0: {'source': 13.0, 'destination': 16.0, 'fiber_length': 102.9498356633842},
    24.0: {'source': 15.0, 'destination': 19.0, 'fiber_length': 98.9837355030138},
    25.0: {'source': 17.0, 'destination': 24.0, 'fiber_length': 210.9792892346443},
    26.0: {'source': 17.0, 'destination': 18.0, 'fiber_length': 120.4791984364237},
    27.0: {'source': 18.0, 'destination': 19.0, 'fiber_length': 100.3611222375359},
    28.0: {'source': 21.0, 'destination': 27.0, 'fiber_length': 223.5563625169812},
    29.0: {'source': 21.0, 'destination': 22.0, 'fiber_length': 243.011643971792},
    30.0: {'source': 22.0, 'destination': 23.0, 'fiber_length': 203.7644228347264},
    31.0: {'source': 25.0, 'destination': 26.0, 'fiber_length': 156.986668574074},
    32.0: {'source': 26.0, 'destination': 27.0, 'fiber_length': 61.91455731777963},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
