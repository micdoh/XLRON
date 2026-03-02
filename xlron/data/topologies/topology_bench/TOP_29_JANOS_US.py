def create_janos_us_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 47.45, 'long': -122.3, 'location': 'SeaTac', 'country': 'United States of America (the)'},
    2: {'lat': 33.93, 'long': -118.4, 'location': 'El Segundo', 'country': 'United States of America (the)'},
    3: {'lat': 37.62, 'long': -122.38, 'location': 'Millbrae', 'country': 'United States of America (the)'},
    4: {'lat': 36.08, 'long': -115.17, 'location': 'Paradise', 'country': 'United States of America (the)'},
    5: {'lat': 40.78, 'long': -111.97, 'location': 'Salt Lake City', 'country': 'United States of America (the)'},
    6: {'lat': 31.8, 'long': -106.4, 'location': 'Fort Bliss', 'country': 'United States of America (the)'},
    7: {'lat': 32.85, 'long': -96.85, 'location': 'University Park', 'country': 'United States of America (the)'},
    8: {'lat': 29.97, 'long': -95.35, 'location': 'Aldine', 'country': 'United States of America (the)'},
    9: {'lat': 36.2, 'long': -95.9, 'location': 'Owasso', 'country': 'United States of America (the)'},
    10: {'lat': 45.07, 'long': -93.38, 'location': 'New Hope', 'country': 'United States of America (the)'},
    11: {'lat': 39.32, 'long': -94.72, 'location': 'Platte City', 'country': 'United States of America (the)'},
    12: {'lat': 39.75, 'long': -104.87, 'location': 'Aurora', 'country': 'United States of America (the)'},
    13: {'lat': 41.98, 'long': -87.9, 'location': 'Rosemont', 'country': 'United States of America (the)'},
    14: {'lat': 39.65, 'long': -86.27, 'location': 'Mooresville', 'country': 'United States of America (the)'},
    15: {'lat': 42.42, 'long': -83.02, 'location': 'Hamtramck', 'country': 'United States of America (the)'},
    16: {'lat': 38.75, 'long': -90.37, 'location': 'Hazelwood', 'country': 'United States of America (the)'},
    17: {'lat': 36.12, 'long': -86.68, 'location': 'Oak Hill', 'country': 'United States of America (the)'},
    18: {'lat': 41.52, 'long': -81.68, 'location': 'Cleveland', 'country': 'United States of America (the)'},
    19: {'lat': 40.65, 'long': -73.78, 'location': 'Inwood', 'country': 'United States of America (the)'},
    20: {'lat': 42.75, 'long': -73.8, 'location': 'Colonie', 'country': 'United States of America (the)'},
    21: {'lat': 35.22, 'long': -80.93, 'location': 'Charlotte', 'country': 'United States of America (the)'},
    22: {'lat': 29.83, 'long': -90.02, 'location': 'Belle Chasse', 'country': 'United States of America (the)'},
    23: {'lat': 42.37, 'long': -71.03, 'location': 'Chelsea', 'country': 'United States of America (the)'},
    24: {'lat': 33.65, 'long': -84.42, 'location': 'Hapeville', 'country': 'United States of America (the)'},
    25: {'lat': 25.82, 'long': -80.28, 'location': 'Miami Springs', 'country': 'United States of America (the)'},
    26: {'lat': 38.85, 'long': -77.04, 'location': 'Washington, D.C.', 'country': 'United States of America (the)'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 3.0, 'fiber_length': 1500.0},
    2.0: {'source': 1.0, 'destination': 5.0, 'fiber_length': 1500.0},
    3.0: {'source': 2.0, 'destination': 3.0, 'fiber_length': 817.6605877784029},
    4.0: {'source': 2.0, 'destination': 4.0, 'fiber_length': 568.5537689973014},
    5.0: {'source': 2.0, 'destination': 6.0, 'fiber_length': 1500.0},
    6.0: {'source': 3.0, 'destination': 5.0, 'fiber_length': 1443.923286951793},
    7.0: {'source': 4.0, 'destination': 5.0, 'fiber_length': 888.2868748003677},
    8.0: {'source': 4.0, 'destination': 6.0, 'fiber_length': 1407.05151144063},
    9.0: {'source': 5.0, 'destination': 12.0, 'fiber_length': 919.5498361442023},
    10.0: {'source': 6.0, 'destination': 7.0, 'fiber_length': 1356.880176796738},
    11.0: {'source': 6.0, 'destination': 8.0, 'fiber_length': 1500.0},
    12.0: {'source': 7.0, 'destination': 8.0, 'fiber_length': 525.6614856052462},
    13.0: {'source': 7.0, 'destination': 9.0, 'fiber_length': 573.791736024592},
    14.0: {'source': 7.0, 'destination': 12.0, 'fiber_length': 1500.0},
    15.0: {'source': 7.0, 'destination': 17.0, 'fiber_length': 1499.83438340173},
    16.0: {'source': 8.0, 'destination': 22.0, 'fiber_length': 770.9584232572195},
    17.0: {'source': 9.0, 'destination': 11.0, 'fiber_length': 543.1408066392901},
    18.0: {'source': 9.0, 'destination': 16.0, 'fiber_length': 846.3620628698229},
    19.0: {'source': 10.0, 'destination': 11.0, 'fiber_length': 973.2043247667134},
    20.0: {'source': 10.0, 'destination': 13.0, 'fiber_length': 839.2186011198173},
    21.0: {'source': 11.0, 'destination': 12.0, 'fiber_length': 1306.925312351537},
    22.0: {'source': 11.0, 'destination': 16.0, 'fiber_length': 571.4797711167189},
    23.0: {'source': 13.0, 'destination': 14.0, 'fiber_length': 439.7124128669452},
    24.0: {'source': 13.0, 'destination': 15.0, 'fiber_length': 607.3384938594429},
    25.0: {'source': 13.0, 'destination': 16.0, 'fiber_length': 623.4452373958061},
    26.0: {'source': 14.0, 'destination': 16.0, 'fiber_length': 550.7361701104992},
    27.0: {'source': 14.0, 'destination': 17.0, 'fiber_length': 591.2435522717495},
    28.0: {'source': 14.0, 'destination': 18.0, 'fiber_length': 659.6582332449932},
    29.0: {'source': 15.0, 'destination': 18.0, 'fiber_length': 223.9302256422749},
    30.0: {'source': 17.0, 'destination': 21.0, 'fiber_length': 793.3267082270377},
    31.0: {'source': 17.0, 'destination': 24.0, 'fiber_length': 515.0705342016395},
    32.0: {'source': 18.0, 'destination': 20.0, 'fiber_length': 995.613671639564},
    33.0: {'source': 18.0, 'destination': 26.0, 'fiber_length': 740.011197632198},
    34.0: {'source': 19.0, 'destination': 20.0, 'fiber_length': 350.2728708484958},
    35.0: {'source': 19.0, 'destination': 23.0, 'fiber_length': 447.4805086812668},
    36.0: {'source': 19.0, 'destination': 26.0, 'fiber_length': 514.6263190806314},
    37.0: {'source': 20.0, 'destination': 23.0, 'fiber_length': 346.1409740121352},
    38.0: {'source': 21.0, 'destination': 24.0, 'fiber_length': 546.8210165297021},
    39.0: {'source': 21.0, 'destination': 26.0, 'fiber_length': 796.5881016659589},
    40.0: {'source': 22.0, 'destination': 24.0, 'fiber_length': 1017.988857051166},
    41.0: {'source': 22.0, 'destination': 25.0, 'fiber_length': 1500.0},
    42.0: {'source': 24.0, 'destination': 25.0, 'fiber_length': 1436.651367091551},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
