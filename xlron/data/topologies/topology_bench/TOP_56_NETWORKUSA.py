def create_networkusa_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 29.95, 'long': -90.08, 'location': 'New Orleans', 'country': 'United States of America (the)'},
    2: {'lat': 29.8, 'long': -90.82, 'location': 'Thibodaux', 'country': 'United States of America (the)'},
    3: {'lat': 30.48, 'long': -90.1, 'location': 'Covington', 'country': 'United States of America (the)'},
    4: {'lat': 30.37, 'long': -89.09, 'location': 'Gulfport', 'country': 'United States of America (the)'},
    5: {'lat': 30.45, 'long': -91.15, 'location': 'Baton Rouge', 'country': 'United States of America (the)'},
    6: {'lat': 30.22, 'long': -92.02, 'location': 'Lafayette', 'country': 'United States of America (the)'},
    7: {'lat': 29.7, 'long': -91.21, 'location': 'Morgan City', 'country': 'United States of America (the)'},
    8: {'lat': 30.5, 'long': -90.46, 'location': 'Hammond', 'country': 'United States of America (the)'},
    9: {'lat': 31.31, 'long': -92.45, 'location': 'Alexandria', 'country': 'United States of America (the)'},
    10: {'lat': 31.53, 'long': -92.95, 'location': 'Vienna Bend', 'country': 'United States of America (the)'},
    11: {'lat': 29.42, 'long': -98.49, 'location': 'San Antonio', 'country': 'United States of America (the)'},
    12: {'lat': 30.27, 'long': -97.74, 'location': 'Austin', 'country': 'United States of America (the)'},
    13: {'lat': 30.63, 'long': -96.33, 'location': 'College Station', 'country': 'United States of America (the)'},
    14: {'lat': 31.55, 'long': -97.15, 'location': 'Waco', 'country': 'United States of America (the)'},
    15: {'lat': 31.76, 'long': -93.09, 'location': 'Natchitoches', 'country': 'United States of America (the)'},
    16: {'lat': 29.76, 'long': -95.36, 'location': 'Houston', 'country': 'United States of America (the)'},
    17: {'lat': 30.06, 'long': -94.8, 'location': 'Liberty', 'country': 'United States of America (the)'},
    18: {'lat': 30.53, 'long': -92.08, 'location': 'Opelousas', 'country': 'United States of America (the)'},
    19: {'lat': 30.84, 'long': -92.26, 'location': 'Bunkie', 'country': 'United States of America (the)'},
    20: {'lat': 31.05, 'long': -93.21, 'location': 'Fort Polk South', 'country': 'United States of America (the)'},
    21: {'lat': 32.01, 'long': -93.34, 'location': 'Coushatta', 'country': 'United States of America (the)'},
    22: {'lat': 30.04, 'long': -94.42, 'location': 'China', 'country': 'United States of America (the)'},
    23: {'lat': 30.09, 'long': -94.1, 'location': 'Beaumont', 'country': 'United States of America (the)'},
    24: {'lat': 30.21, 'long': -93.2, 'location': 'Lake Charles', 'country': 'United States of America (the)'},
    25: {'lat': 30.85, 'long': -93.29, 'location': 'DeRidder', 'country': 'United States of America (the)'},
    26: {'lat': 32.3, 'long': -90.18, 'location': 'Jackson', 'country': 'United States of America (the)'},
    27: {'lat': 30.28, 'long': -89.78, 'location': 'Slidell', 'country': 'United States of America (the)'},
    28: {'lat': 32.78, 'long': -96.81, 'location': 'Dallas', 'country': 'United States of America (the)'},
    29: {'lat': 32.35, 'long': -95.3, 'location': 'Tyler', 'country': 'United States of America (the)'},
    30: {'lat': 32.5, 'long': -94.74, 'location': 'Longview', 'country': 'United States of America (the)'},
    31: {'lat': 32.53, 'long': -93.75, 'location': 'Shreveport', 'country': 'United States of America (the)'},
    32: {'lat': 32.52, 'long': -93.73, 'location': 'Bossier City', 'country': 'United States of America (the)'},
    33: {'lat': 32.62, 'long': -93.29, 'location': 'Minden', 'country': 'United States of America (the)'},
    34: {'lat': 32.52, 'long': -92.64, 'location': 'Ruston', 'country': 'United States of America (the)'},
    35: {'lat': 32.51, 'long': -92.12, 'location': 'Monroe', 'country': 'United States of America (the)'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 109.9099113503352},
    2.0: {'source': 1.0, 'destination': 4.0, 'fiber_length': 159.0301992786792},
    3.0: {'source': 1.0, 'destination': 8.0, 'fiber_length': 106.839090890624},
    4.0: {'source': 2.0, 'destination': 7.0, 'fiber_length': 58.88695641632107},
    5.0: {'source': 3.0, 'destination': 27.0, 'fiber_length': 56.85869217629584},
    6.0: {'source': 3.0, 'destination': 8.0, 'fiber_length': 51.84946378048295},
    7.0: {'source': 4.0, 'destination': 27.0, 'fiber_length': 100.4676517868845},
    8.0: {'source': 5.0, 'destination': 6.0, 'fiber_length': 130.9851627109675},
    9.0: {'source': 5.0, 'destination': 8.0, 'fiber_length': 99.53739324889001},
    10.0: {'source': 6.0, 'destination': 18.0, 'fiber_length': 52.42152540582747},
    11.0: {'source': 6.0, 'destination': 7.0, 'fiber_length': 145.6798308289586},
    12.0: {'source': 6.0, 'destination': 24.0, 'fiber_length': 170.0837548602486},
    13.0: {'source': 8.0, 'destination': 26.0, 'fiber_length': 302.8606396900053},
    14.0: {'source': 9.0, 'destination': 10.0, 'fiber_length': 80.07056650096938},
    15.0: {'source': 9.0, 'destination': 19.0, 'fiber_length': 82.9583652662024},
    16.0: {'source': 10.0, 'destination': 15.0, 'fiber_length': 43.20689542819208},
    17.0: {'source': 11.0, 'destination': 12.0, 'fiber_length': 178.528161273931},
    18.0: {'source': 11.0, 'destination': 16.0, 'fiber_length': 457.4870157403198},
    19.0: {'source': 12.0, 'destination': 13.0, 'fiber_length': 211.442669539775},
    20.0: {'source': 13.0, 'destination': 14.0, 'fiber_length': 193.0385827633951},
    21.0: {'source': 14.0, 'destination': 28.0, 'fiber_length': 210.6959072219843},
    22.0: {'source': 15.0, 'destination': 21.0, 'fiber_length': 54.70219883476837},
    23.0: {'source': 16.0, 'destination': 17.0, 'fiber_length': 95.17763035009355},
    24.0: {'source': 17.0, 'destination': 22.0, 'fiber_length': 54.9632714608842},
    25.0: {'source': 18.0, 'destination': 19.0, 'fiber_length': 57.79353400373419},
    26.0: {'source': 20.0, 'destination': 25.0, 'fiber_length': 35.26672267006701},
    27.0: {'source': 21.0, 'destination': 31.0, 'fiber_length': 104.2391761962202},
    28.0: {'source': 22.0, 'destination': 23.0, 'fiber_length': 46.93932264556766},
    29.0: {'source': 23.0, 'destination': 24.0, 'fiber_length': 131.3385144914388},
    30.0: {'source': 24.0, 'destination': 25.0, 'fiber_length': 107.5273736163993},
    31.0: {'source': 26.0, 'destination': 31.0, 'fiber_length': 504.1082021016392},
    32.0: {'source': 28.0, 'destination': 29.0, 'fiber_length': 224.0467180949339},
    33.0: {'source': 29.0, 'destination': 30.0, 'fiber_length': 82.71583618424468},
    34.0: {'source': 30.0, 'destination': 31.0, 'fiber_length': 139.3307052177493},
    35.0: {'source': 31.0, 'destination': 35.0, 'fiber_length': 229.26520678719},
    36.0: {'source': 31.0, 'destination': 32.0, 'fiber_length': 3.270004842872805},
    37.0: {'source': 32.0, 'destination': 33.0, 'fiber_length': 64.05667894861249},
    38.0: {'source': 33.0, 'destination': 34.0, 'fiber_length': 92.87489744679914},
    39.0: {'source': 34.0, 'destination': 35.0, 'fiber_length': 73.15580406714743},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
