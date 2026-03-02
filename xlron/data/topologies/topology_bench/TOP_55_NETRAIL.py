def create_netrail_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 37.44, 'long': -122.14, 'location': 'Palo Alto', 'country': 'United States of America (the)'},
    2: {'lat': 41.85, 'long': -87.65, 'location': 'Chicago', 'country': 'United States of America (the)'},
    3: {'lat': 40.71, 'long': -74.01, 'location': 'New York City', 'country': 'United States of America (the)'},
    4: {'lat': 39.29, 'long': -76.61, 'location': 'Baltimore', 'country': 'United States of America (the)'},
    5: {'lat': 38.9, 'long': -77.04, 'location': 'Washington, D.C.', 'country': 'United States of America (the)'},
    6: {'lat': 25.77, 'long': -80.19, 'location': 'Miami', 'country': 'United States of America (the)'},
    7: {'lat': 33.75, 'long': -84.39, 'location': 'Atlanta', 'country': 'United States of America (the)'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 5.0, 'fiber_length': 4881.744183715728},
    2.0: {'source': 1.0, 'destination': 7.0, 'fiber_length': 4268.785450169966},
    3.0: {'source': 2.0, 'destination': 3.0, 'fiber_length': 1500.0},
    4.0: {'source': 2.0, 'destination': 7.0, 'fiber_length': 1417.377937950516},
    5.0: {'source': 3.0, 'destination': 4.0, 'fiber_length': 407.9571918717692},
    6.0: {'source': 3.0, 'destination': 5.0, 'fiber_length': 491.7553096743694},
    7.0: {'source': 4.0, 'destination': 5.0, 'fiber_length': 85.61340860642284},
    8.0: {'source': 5.0, 'destination': 6.0, 'fiber_length': 1861.74600958398},
    9.0: {'source': 5.0, 'destination': 7.0, 'fiber_length': 1308.123444881084},
    10.0: {'source': 6.0, 'destination': 7.0, 'fiber_length': 1462.933348688248},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
