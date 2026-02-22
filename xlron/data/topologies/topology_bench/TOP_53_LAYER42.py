def create_layer42_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 47.61, 'long': -122.33, 'location': 'Seattle', 'country': 'United States of America (the)'},
    2: {'lat': 37.77, 'long': -122.42, 'location': 'San Francisco', 'country': 'United States of America (the)'},
    3: {'lat': 34.05, 'long': -118.24, 'location': 'Los Angeles', 'country': 'United States of America (the)'},
    4: {'lat': 41.85, 'long': -87.65, 'location': 'Chicago', 'country': 'United States of America (the)'},
    5: {'lat': 40.71, 'long': -74.01, 'location': 'New York City', 'country': 'United States of America (the)'},
    6: {'lat': 38.9, 'long': -77.04, 'location': 'Washington, D.C.', 'country': 'United States of America (the)'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 1500.0},
    2.0: {'source': 2.0, 'destination': 3.0, 'fiber_length': 838.7553033988663},
    3.0: {'source': 2.0, 'destination': 4.0, 'fiber_length': 3729.536934968901},
    4.0: {'source': 2.0, 'destination': 6.0, 'fiber_length': 4898.330764100102},
    5.0: {'source': 4.0, 'destination': 5.0, 'fiber_length': 1500.0},
    6.0: {'source': 4.0, 'destination': 6.0, 'fiber_length': 1433.948068913156},
    7.0: {'source': 5.0, 'destination': 6.0, 'fiber_length': 491.7553096743694},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
