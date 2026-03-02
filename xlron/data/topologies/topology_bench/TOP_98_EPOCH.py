def create_epoch_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 37.44, 'long': -122.14, 'location': 'Palo Alto', 'country': 'United States of America (the)'},
    2: {'lat': 34.05, 'long': -118.24, 'location': 'Los Angeles', 'country': 'United States of America (the)'},
    3: {'lat': 39.74, 'long': -104.98, 'location': 'Denver', 'country': 'United States of America (the)'},
    4: {'lat': 41.85, 'long': -87.65, 'location': 'Chicago', 'country': 'United States of America (the)'},
    5: {'lat': 38.9, 'long': -77.27, 'location': 'Vienna', 'country': 'United States of America (the)'},
    6: {'lat': 33.75, 'long': -84.39, 'location': 'Atlanta', 'country': 'United States of America (the)'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 773.4325645850809},
    2.0: {'source': 1.0, 'destination': 3.0, 'fiber_length': 1888.454439970209},
    3.0: {'source': 1.0, 'destination': 5.0, 'fiber_length': 4857.418818853637},
    4.0: {'source': 2.0, 'destination': 6.0, 'fiber_length': 3887.291603216945},
    5.0: {'source': 3.0, 'destination': 4.0, 'fiber_length': 1843.716434385458},
    6.0: {'source': 4.0, 'destination': 5.0, 'fiber_length': 1406.590956106817},
    7.0: {'source': 5.0, 'destination': 6.0, 'fiber_length': 1285.010042656808},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
