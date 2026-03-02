def create_gts_czech_republic_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 49.4, 'long': 13.3, 'location': 'Klatovy', 'country': 'Czechia'},
    2: {'lat': 49.31, 'long': 14.15, 'location': 'Pisek', 'country': 'Czechia'},
    3: {'lat': 50.46, 'long': 13.42, 'location': 'Chomutov', 'country': 'Czechia'},
    4: {'lat': 49.75, 'long': 13.38, 'location': 'Pilsen', 'country': 'Czechia'},
    5: {'lat': 49.14, 'long': 15.0, 'location': 'Jindrichuv Hradec', 'country': 'Czechia'},
    6: {'lat': 49.61, 'long': 15.58, 'location': 'Havlickuv Brod', 'country': 'Czechia'},
    7: {'lat': 48.97, 'long': 14.47, 'location': 'Ceske Budejovice', 'country': 'Czechia'},
    8: {'lat': 49.41, 'long': 14.66, 'location': 'Tabor', 'country': 'Czechia'},
    9: {'lat': 49.2, 'long': 16.61, 'location': 'Brno', 'country': 'Czechia'},
    10: {'lat': 50.66, 'long': 14.03, 'location': 'Usti nad Labem', 'country': 'Czechia'},
    11: {'lat': 50.03, 'long': 15.2, 'location': 'Kolin', 'country': 'Czechia'},
    12: {'lat': 49.83, 'long': 18.28, 'location': 'Ostrava', 'country': 'Czechia'},
    13: {'lat': 49.22, 'long': 17.67, 'location': 'Zlin', 'country': 'Czechia'},
    14: {'lat': 49.03, 'long': 15.5, 'location': 'Jemnice', 'country': 'Czechia'},
    15: {'lat': 48.85, 'long': 17.13, 'location': 'Hodonin', 'country': 'Czechia'},
    16: {'lat': 49.47, 'long': 17.11, 'location': 'Prostejov', 'country': 'Czechia'},
    17: {'lat': 49.96, 'long': 14.07, 'location': 'Beroun', 'country': 'Czechia'},
    18: {'lat': 50.23, 'long': 12.87, 'location': 'Karlovy Vary', 'country': 'Czechia'},
    19: {'lat': 49.94, 'long': 17.9, 'location': 'Opava', 'country': 'Czechia'},
    20: {'lat': 49.6, 'long': 17.25, 'location': 'Olomouc', 'country': 'Czechia'},
    21: {'lat': 49.9, 'long': 16.44, 'location': 'Ceska Trebova', 'country': 'Czechia'},
    22: {'lat': 50.77, 'long': 15.06, 'location': 'Liberec', 'country': 'Czechia'},
    23: {'lat': 50.6, 'long': 15.34, 'location': 'Semily', 'country': 'Czechia'},
    24: {'lat': 50.15, 'long': 14.1, 'location': 'Kladno', 'country': 'Czechia'},
    25: {'lat': 50.09, 'long': 14.42, 'location': 'Prague', 'country': 'Czechia'},
    26: {'lat': 50.41, 'long': 14.9, 'location': 'Mlada Boleslav', 'country': 'Czechia'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 4.0, 'fiber_length': 59.01507299542541},
    2.0: {'source': 2.0, 'destination': 7.0, 'fiber_length': 66.59716207899825},
    3.0: {'source': 3.0, 'destination': 10.0, 'fiber_length': 72.73484848207556},
    4.0: {'source': 3.0, 'destination': 18.0, 'fiber_length': 69.99170242282302},
    5.0: {'source': 4.0, 'destination': 17.0, 'fiber_length': 82.0505994522759},
    6.0: {'source': 4.0, 'destination': 18.0, 'fiber_length': 96.95617952126375},
    7.0: {'source': 5.0, 'destination': 6.0, 'fiber_length': 100.5618534474931},
    8.0: {'source': 5.0, 'destination': 7.0, 'fiber_length': 64.4982574068788},
    9.0: {'source': 6.0, 'destination': 9.0, 'fiber_length': 131.0451677159967},
    10.0: {'source': 7.0, 'destination': 25.0, 'fiber_length': 186.8858670613935},
    11.0: {'source': 7.0, 'destination': 8.0, 'fiber_length': 76.25514690688053},
    12.0: {'source': 9.0, 'destination': 15.0, 'fiber_length': 81.50069339341765},
    13.0: {'source': 10.0, 'destination': 22.0, 'fiber_length': 110.3130666224108},
    14.0: {'source': 11.0, 'destination': 25.0, 'fiber_length': 84.11803537378353},
    15.0: {'source': 12.0, 'destination': 19.0, 'fiber_length': 44.76998048270598},
    16.0: {'source': 12.0, 'destination': 20.0, 'fiber_length': 117.5181952711605},
    17.0: {'source': 12.0, 'destination': 13.0, 'fiber_length': 121.2979327333021},
    18.0: {'source': 13.0, 'destination': 14.0, 'fiber_length': 238.9592328689168},
    19.0: {'source': 14.0, 'destination': 15.0, 'fiber_length': 181.0809673157038},
    20.0: {'source': 16.0, 'destination': 20.0, 'fiber_length': 26.4538710748345},
    21.0: {'source': 17.0, 'destination': 25.0, 'fiber_length': 43.32148031128548},
    22.0: {'source': 20.0, 'destination': 21.0, 'fiber_length': 100.6160808488183},
    23.0: {'source': 22.0, 'destination': 23.0, 'fiber_length': 40.98201015679786},
    24.0: {'source': 24.0, 'destination': 25.0, 'fiber_length': 35.65535993070064},
    25.0: {'source': 25.0, 'destination': 26.0, 'fiber_length': 73.95598548369489},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
