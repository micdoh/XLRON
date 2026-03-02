def create_elibackbone_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 36.17, 'long': -115.14, 'location': 'Las Vegas', 'country': 'United States of America (the)'},
    2: {'lat': 34.05, 'long': -118.24, 'location': 'Los Angeles', 'country': 'United States of America (the)'},
    3: {'lat': 32.78, 'long': -96.81, 'location': 'Dallas', 'country': 'United States of America (the)'},
    4: {'lat': 33.45, 'long': -112.07, 'location': 'Phoenix', 'country': 'United States of America (the)'},
    5: {'lat': 37.35, 'long': -121.96, 'location': 'Santa Clara', 'country': 'United States of America (the)'},
    6: {'lat': 45.52, 'long': -122.68, 'location': 'Portland', 'country': 'United States of America (the)'},
    7: {'lat': 38.58, 'long': -121.49, 'location': 'Sacramento', 'country': 'United States of America (the)'},
    8: {'lat': 37.44, 'long': -122.14, 'location': 'Palo Alto', 'country': 'United States of America (the)'},
    9: {'lat': 47.25, 'long': -122.44, 'location': 'Tacoma', 'country': 'United States of America (the)'},
    10: {'lat': 47.61, 'long': -122.33, 'location': 'Seattle', 'country': 'United States of America (the)'},
    11: {'lat': 33.75, 'long': -84.39, 'location': 'Atlanta', 'country': 'United States of America (the)'},
    12: {'lat': 29.76, 'long': -95.36, 'location': 'Houston', 'country': 'United States of America (the)'},
    13: {'lat': 47.66, 'long': -117.43, 'location': 'Spokane', 'country': 'United States of America (the)'},
    14: {'lat': 43.61, 'long': -116.2, 'location': 'Boise', 'country': 'United States of America (the)'},
    15: {'lat': 40.76, 'long': -111.89, 'location': 'Salt Lake City', 'country': 'United States of America (the)'},
    16: {'lat': 44.98, 'long': -93.26, 'location': 'Minneapolis', 'country': 'United States of America (the)'},
    17: {'lat': 41.85, 'long': -87.65, 'location': 'Chicago', 'country': 'United States of America (the)'},
    18: {'lat': 43.15, 'long': -77.62, 'location': 'Rochester', 'country': 'United States of America (the)'},
    19: {'lat': 40.74, 'long': -74.17, 'location': 'Newark', 'country': 'United States of America (the)'},
    20: {'lat': 38.9, 'long': -77.04, 'location': 'Washington, D.C.', 'country': 'United States of America (the)'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 551.2511575012703},
    2.0: {'source': 1.0, 'destination': 4.0, 'fiber_length': 618.4493777535091},
    3.0: {'source': 1.0, 'destination': 15.0, 'fiber_length': 875.1907174327788},
    4.0: {'source': 2.0, 'destination': 4.0, 'fiber_length': 861.3692628497527},
    5.0: {'source': 2.0, 'destination': 5.0, 'fiber_length': 746.0789546600618},
    6.0: {'source': 3.0, 'destination': 17.0, 'fiber_length': 1614.90805684733},
    7.0: {'source': 3.0, 'destination': 12.0, 'fiber_length': 544.4600499700148},
    8.0: {'source': 3.0, 'destination': 4.0, 'fiber_length': 1777.377955221904},
    9.0: {'source': 3.0, 'destination': 20.0, 'fiber_length': 2378.380847303889},
    10.0: {'source': 5.0, 'destination': 8.0, 'fiber_length': 28.1825717828491},
    11.0: {'source': 6.0, 'destination': 9.0, 'fiber_length': 289.8686285600951},
    12.0: {'source': 6.0, 'destination': 7.0, 'fiber_length': 1166.845178904244},
    13.0: {'source': 7.0, 'destination': 9.0, 'fiber_length': 1450.706802327106},
    14.0: {'source': 7.0, 'destination': 15.0, 'fiber_length': 1284.236403092044},
    15.0: {'source': 7.0, 'destination': 8.0, 'fiber_length': 208.4477088878102},
    16.0: {'source': 8.0, 'destination': 10.0, 'fiber_length': 1500.0},
    17.0: {'source': 9.0, 'destination': 10.0, 'fiber_length': 61.31460492016745},
    18.0: {'source': 10.0, 'destination': 13.0, 'fiber_length': 550.6983275646231},
    19.0: {'source': 10.0, 'destination': 14.0, 'fiber_length': 977.5315001175619},
    20.0: {'source': 10.0, 'destination': 15.0, 'fiber_length': 1500.0},
    21.0: {'source': 10.0, 'destination': 16.0, 'fiber_length': 2798.902600946488},
    22.0: {'source': 11.0, 'destination': 12.0, 'fiber_length': 1500.0},
    23.0: {'source': 11.0, 'destination': 20.0, 'fiber_length': 1308.123444881084},
    24.0: {'source': 13.0, 'destination': 14.0, 'fiber_length': 690.546588040392},
    25.0: {'source': 14.0, 'destination': 15.0, 'fiber_length': 713.7523611067775},
    26.0: {'source': 15.0, 'destination': 17.0, 'fiber_length': 2527.089055227165},
    27.0: {'source': 16.0, 'destination': 17.0, 'fiber_length': 856.6967858075002},
    28.0: {'source': 17.0, 'destination': 18.0, 'fiber_length': 1251.524923458717},
    29.0: {'source': 18.0, 'destination': 19.0, 'fiber_length': 587.0659618990562},
    30.0: {'source': 19.0, 'destination': 20.0, 'fiber_length': 478.870671721509},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
