def create_gblnet_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 53.08, 'long': -0.14, 'location': 'Coningsby', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    2: {'lat': 51.51, 'long': -0.13, 'location': 'London', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    3: {'lat': 59.33, 'long': 18.06, 'location': 'Stockholm', 'country': 'Sweden'},
    4: {'lat': 59.89, 'long': 30.26, 'location': 'Avtovo', 'country': 'Russian Federation (the)'},
    5: {'lat': 55.75, 'long': 37.62, 'location': 'Moscow', 'country': 'Russian Federation (the)'},
    6: {'lat': 52.37, 'long': 4.89, 'location': 'Amsterdam', 'country': 'Netherlands (Kingdom of the)'},
    7: {'lat': 50.12, 'long': 8.68, 'location': 'Frankfurt am Main', 'country': 'Germany'},
    8: {'lat': 48.85, 'long': 2.35, 'location': 'Paris', 'country': 'France'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 6.0, 'fiber_length': 521.6095788482335},
    2.0: {'source': 2.0, 'destination': 6.0, 'fiber_length': 535.6163051270169},
    3.0: {'source': 3.0, 'destination': 4.0, 'fiber_length': 1032.15219238748},
    4.0: {'source': 3.0, 'destination': 6.0, 'fiber_length': 1500.0},
    5.0: {'source': 4.0, 'destination': 5.0, 'fiber_length': 949.8667213173954},
    6.0: {'source': 6.0, 'destination': 7.0, 'fiber_length': 545.2339470871901},
    7.0: {'source': 6.0, 'destination': 8.0, 'fiber_length': 645.6357804808555},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
