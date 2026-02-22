def create_getnet_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 47.61, 'long': -122.33, 'location': 'Seattle', 'country': 'United States of America (the)'},
    2: {'lat': 37.35, 'long': -121.96, 'location': 'Santa Clara', 'country': 'United States of America (the)'},
    3: {'lat': 33.45, 'long': -112.07, 'location': 'Phoenix', 'country': 'United States of America (the)'},
    4: {'lat': 32.22, 'long': -110.93, 'location': 'Tucson', 'country': 'United States of America (the)'},
    5: {'lat': 38.9, 'long': -77.04, 'location': 'Washington, D.C.', 'country': 'United States of America (the)'},
    6: {'lat': 39.29, 'long': -76.61, 'location': 'Baltimore', 'country': 'United States of America (the)'},
    7: {'lat': 40.44, 'long': -80.0, 'location': 'Pittsburgh', 'country': 'United States of America (the)'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 1500.0},
    2.0: {'source': 2.0, 'destination': 3.0, 'fiber_length': 1492.603044009537},
    3.0: {'source': 2.0, 'destination': 5.0, 'fiber_length': 4866.289572772755},
    4.0: {'source': 2.0, 'destination': 7.0, 'fiber_length': 4516.783096637158},
    5.0: {'source': 3.0, 'destination': 4.0, 'fiber_length': 260.0209332979543},
    6.0: {'source': 3.0, 'destination': 5.0, 'fiber_length': 3977.389262103784},
    7.0: {'source': 5.0, 'destination': 6.0, 'fiber_length': 85.61340860642284},
    8.0: {'source': 6.0, 'destination': 7.0, 'fiber_length': 474.4497917256419},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
