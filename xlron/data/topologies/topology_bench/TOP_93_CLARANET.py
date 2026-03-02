def create_claranet_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 37.02, 'long': -7.93, 'location': 'Faro', 'country': 'Portugal'},
    2: {'lat': 40.42, 'long': -3.7, 'location': 'City Center', 'country': 'Spain'},
    3: {'lat': 41.15, 'long': -8.62, 'location': 'Porto', 'country': 'Portugal'},
    4: {'lat': 38.72, 'long': -9.13, 'location': 'Lisbon', 'country': 'Portugal'},
    5: {'lat': 41.39, 'long': 2.16, 'location': 'Barcelona', 'country': 'Spain'},
    6: {'lat': 53.48, 'long': -2.24, 'location': 'Manchester', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    7: {'lat': 53.08, 'long': -0.14, 'location': 'Coningsby', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    8: {'lat': 52.37, 'long': 4.89, 'location': 'Amsterdam', 'country': 'Netherlands (Kingdom of the)'},
    9: {'lat': 51.44, 'long': 5.48, 'location': 'Eindhoven', 'country': 'Netherlands (Kingdom of the)'},
    10: {'lat': 52.52, 'long': 13.41, 'location': 'Berlin', 'country': 'Germany'},
    11: {'lat': 50.12, 'long': 8.68, 'location': 'Frankfurt am Main', 'country': 'Germany'},
    12: {'lat': 48.14, 'long': 11.58, 'location': 'Munich', 'country': 'Germany'},
    13: {'lat': 48.85, 'long': 2.35, 'location': 'Paris', 'country': 'France'},
    14: {'lat': 48.08, 'long': -1.68, 'location': 'Rennes', 'country': 'France'},
    15: {'lat': 51.51, 'long': -0.13, 'location': 'London', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 4.0, 'fiber_length': 324.5882130963141},
    2.0: {'source': 2.0, 'destination': 4.0, 'fiber_length': 753.3558347662502},
    3.0: {'source': 2.0, 'destination': 5.0, 'fiber_length': 756.0643238408899},
    4.0: {'source': 3.0, 'destination': 4.0, 'fiber_length': 410.517782476493},
    5.0: {'source': 4.0, 'destination': 15.0, 'fiber_length': 1981.348079362193},
    6.0: {'source': 5.0, 'destination': 13.0, 'fiber_length': 1244.471006894535},
    7.0: {'source': 6.0, 'destination': 15.0, 'fiber_length': 392.2307757921959},
    8.0: {'source': 7.0, 'destination': 15.0, 'fiber_length': 261.8660383917509},
    9.0: {'source': 8.0, 'destination': 9.0, 'fiber_length': 166.5742427087172},
    10.0: {'source': 8.0, 'destination': 11.0, 'fiber_length': 545.2339470871901},
    11.0: {'source': 8.0, 'destination': 15.0, 'fiber_length': 535.6163051270169},
    12.0: {'source': 10.0, 'destination': 11.0, 'fiber_length': 634.8701531428853},
    13.0: {'source': 10.0, 'destination': 12.0, 'fiber_length': 756.0218233393591},
    14.0: {'source': 11.0, 'destination': 12.0, 'fiber_length': 457.3596175321572},
    15.0: {'source': 11.0, 'destination': 13.0, 'fiber_length': 717.5980194901786},
    16.0: {'source': 11.0, 'destination': 15.0, 'fiber_length': 956.285408716683},
    17.0: {'source': 13.0, 'destination': 14.0, 'fiber_length': 463.7727051434647},
    18.0: {'source': 13.0, 'destination': 15.0, 'fiber_length': 516.656454570689},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
