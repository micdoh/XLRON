def create_darkstrand_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 36.15, 'long': -95.99, 'location': 'Tulsa', 'country': 'United States of America (the)'},
    2: {'lat': 39.11, 'long': -94.63, 'location': 'Kansas City', 'country': 'United States of America (the)'},
    3: {'lat': 29.42, 'long': -98.49, 'location': 'San Antonio', 'country': 'United States of America (the)'},
    4: {'lat': 32.78, 'long': -96.81, 'location': 'Dallas', 'country': 'United States of America (the)'},
    5: {'lat': 30.42, 'long': -87.22, 'location': 'Pensacola', 'country': 'United States of America (the)'},
    6: {'lat': 30.33, 'long': -81.66, 'location': 'Jacksonville', 'country': 'United States of America (the)'},
    7: {'lat': 29.76, 'long': -95.36, 'location': 'Houston', 'country': 'United States of America (the)'},
    8: {'lat': 30.45, 'long': -91.15, 'location': 'Baton Rouge', 'country': 'United States of America (the)'},
    9: {'lat': 33.75, 'long': -84.39, 'location': 'Atlanta', 'country': 'United States of America (the)'},
    10: {'lat': 35.77, 'long': -78.64, 'location': 'Raleigh', 'country': 'United States of America (the)'},
    11: {'lat': 40.71, 'long': -74.01, 'location': 'New York City', 'country': 'United States of America (the)'},
    12: {'lat': 43.05, 'long': -76.15, 'location': 'Syracuse', 'country': 'United States of America (the)'},
    13: {'lat': 41.5, 'long': -81.7, 'location': 'Cleveland', 'country': 'United States of America (the)'},
    14: {'lat': 41.85, 'long': -87.65, 'location': 'Chicago', 'country': 'United States of America (the)'},
    15: {'lat': 36.9, 'long': -104.44, 'location': 'Raton', 'country': 'United States of America (the)'},
    16: {'lat': 40.44, 'long': -80.0, 'location': 'Pittsburgh', 'country': 'United States of America (the)'},
    17: {'lat': 38.9, 'long': -77.04, 'location': 'Washington, D.C.', 'country': 'United States of America (the)'},
    18: {'lat': 39.95, 'long': -75.16, 'location': 'Philadelphia', 'country': 'United States of America (the)'},
    19: {'lat': 34.05, 'long': -118.24, 'location': 'Los Angeles', 'country': 'United States of America (the)'},
    20: {'lat': 31.76, 'long': -106.49, 'location': 'El Paso', 'country': 'United States of America (the)'},
    21: {'lat': 47.61, 'long': -122.33, 'location': 'Seattle', 'country': 'United States of America (the)'},
    22: {'lat': 45.52, 'long': -122.68, 'location': 'Portland', 'country': 'United States of America (the)'},
    23: {'lat': 37.37, 'long': -122.04, 'location': 'Sunnyvale', 'country': 'United States of America (the)'},
    24: {'lat': 43.61, 'long': -116.2, 'location': 'Boise', 'country': 'United States of America (the)'},
    25: {'lat': 40.76, 'long': -111.89, 'location': 'Salt Lake City', 'country': 'United States of America (the)'},
    26: {'lat': 39.74, 'long': -104.98, 'location': 'Denver', 'country': 'United States of America (the)'},
    27: {'lat': 35.08, 'long': -106.65, 'location': 'Albuquerque', 'country': 'United States of America (the)'},
    28: {'lat': 33.45, 'long': -112.07, 'location': 'Phoenix', 'country': 'United States of America (the)'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 525.3552894927698},
    2.0: {'source': 1.0, 'destination': 4.0, 'fiber_length': 573.2818737029461},
    3.0: {'source': 2.0, 'destination': 26.0, 'fiber_length': 1336.875843009606},
    4.0: {'source': 2.0, 'destination': 14.0, 'fiber_length': 996.0670010192133},
    5.0: {'source': 3.0, 'destination': 20.0, 'fiber_length': 1212.754078757391},
    6.0: {'source': 3.0, 'destination': 7.0, 'fiber_length': 457.4870157403198},
    7.0: {'source': 4.0, 'destination': 7.0, 'fiber_length': 544.4600499700148},
    8.0: {'source': 5.0, 'destination': 6.0, 'fiber_length': 800.1306068813958},
    9.0: {'source': 5.0, 'destination': 8.0, 'fiber_length': 565.1635832158854},
    10.0: {'source': 6.0, 'destination': 9.0, 'fiber_length': 688.6727439710183},
    11.0: {'source': 7.0, 'destination': 8.0, 'fiber_length': 618.2396065565714},
    12.0: {'source': 9.0, 'destination': 10.0, 'fiber_length': 856.7329203000172},
    13.0: {'source': 10.0, 'destination': 17.0, 'fiber_length': 563.5055664441111},
    14.0: {'source': 11.0, 'destination': 18.0, 'fiber_length': 193.5165266446945},
    15.0: {'source': 11.0, 'destination': 12.0, 'fiber_length': 472.1418119601072},
    16.0: {'source': 12.0, 'destination': 13.0, 'fiber_length': 731.9310690917647},
    17.0: {'source': 13.0, 'destination': 14.0, 'fiber_length': 743.4071200946746},
    18.0: {'source': 13.0, 'destination': 16.0, 'fiber_length': 277.6478760498783},
    19.0: {'source': 15.0, 'destination': 26.0, 'fiber_length': 478.9289964937095},
    20.0: {'source': 15.0, 'destination': 27.0, 'fiber_length': 425.5352233146512},
    21.0: {'source': 16.0, 'destination': 17.0, 'fiber_length': 458.6437394496578},
    22.0: {'source': 17.0, 'destination': 18.0, 'fiber_length': 298.8881926608235},
    23.0: {'source': 19.0, 'destination': 28.0, 'fiber_length': 861.3692628497527},
    24.0: {'source': 19.0, 'destination': 23.0, 'fiber_length': 755.8274651640023},
    25.0: {'source': 20.0, 'destination': 28.0, 'fiber_length': 833.0157749534733},
    26.0: {'source': 20.0, 'destination': 27.0, 'fiber_length': 554.1982489372652},
    27.0: {'source': 21.0, 'destination': 22.0, 'fiber_length': 350.8979570264834},
    28.0: {'source': 21.0, 'destination': 24.0, 'fiber_length': 977.5315001175619},
    29.0: {'source': 22.0, 'destination': 23.0, 'fiber_length': 1361.697714986217},
    30.0: {'source': 24.0, 'destination': 25.0, 'fiber_length': 713.7523611067775},
    31.0: {'source': 25.0, 'destination': 26.0, 'fiber_length': 895.6985411380621},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
