def create_kentmanjul2005_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 51.33, 'long': 0.5, 'location': 'Aylesford', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    2: {'lat': 51.18, 'long': 0.94, 'location': 'Wye', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    3: {'lat': 51.29, 'long': 0.98, 'location': 'Dunkirk', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    4: {'lat': 51.4, 'long': 0.43, 'location': 'Cuxton', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    5: {'lat': 51.28, 'long': 1.08, 'location': 'Canterbury', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    6: {'lat': 51.38, 'long': 0.53, 'location': 'Chatham', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    7: {'lat': 51.08, 'long': 0.53, 'location': 'Cranbrook', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    8: {'lat': 51.18, 'long': 0.94, 'location': 'Wye', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    9: {'lat': 51.45, 'long': 0.08, 'location': 'Welling', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    10: {'lat': 51.28, 'long': 1.09, 'location': 'Canterbury', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    11: {'lat': 51.15, 'long': 0.24, 'location': 'Royal Tunbridge Wells', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    12: {'lat': 51.36, 'long': 1.41, 'location': 'Ramsgate', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    13: {'lat': 51.38, 'long': 0.51, 'location': 'Rochester', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    14: {'lat': 51.28, 'long': 1.09, 'location': 'Canterbury', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    15: {'lat': 51.27, 'long': 0.5, 'location': 'Maidstone', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    16: {'lat': 51.18, 'long': 0.28, 'location': 'Tonbridge', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 4.0, 'fiber_length': 13.7642876403916},
    2.0: {'source': 1.0, 'destination': 6.0, 'fiber_length': 8.905827237171904},
    3.0: {'source': 1.0, 'destination': 7.0, 'fiber_length': 41.8157833272339},
    4.0: {'source': 1.0, 'destination': 9.0, 'fiber_length': 48.07821096666471},
    5.0: {'source': 1.0, 'destination': 15.0, 'fiber_length': 10.00754339800966},
    6.0: {'source': 1.0, 'destination': 16.0, 'fiber_length': 33.96097955836591},
    7.0: {'source': 2.0, 'destination': 5.0, 'fiber_length': 22.18122978453439},
    8.0: {'source': 2.0, 'destination': 8.0, 'fiber_length': 0.0},
    9.0: {'source': 3.0, 'destination': 4.0, 'fiber_length': 60.16635359396744},
    10.0: {'source': 3.0, 'destination': 5.0, 'fiber_length': 10.56447543922318},
    11.0: {'source': 4.0, 'destination': 9.0, 'fiber_length': 37.34359419267392},
    12.0: {'source': 4.0, 'destination': 13.0, 'fiber_length': 8.969854855057697},
    13.0: {'source': 5.0, 'destination': 12.0, 'fiber_length': 36.89653311964919},
    14.0: {'source': 5.0, 'destination': 14.0, 'fiber_length': 1.043311483748465},
    15.0: {'source': 5.0, 'destination': 10.0, 'fiber_length': 1.043311483748465},
    16.0: {'source': 7.0, 'destination': 8.0, 'fiber_length': 46.04251457957461},
    17.0: {'source': 11.0, 'destination': 16.0, 'fiber_length': 6.522342469188237},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
