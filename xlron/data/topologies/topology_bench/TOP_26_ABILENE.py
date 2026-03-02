def create_abilene_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 33.75, 'long': -84.3833, 'location': 'Atlanta', 'country': 'United States of America (the)'},
    2: {'lat': 34.5, 'long': -85.5, 'location': 'Summerville', 'country': 'United States of America (the)'},
    3: {'lat': 41.8333, 'long': -87.6167, 'location': 'Chicago', 'country': 'United States of America (the)'},
    4: {'lat': 40.75, 'long': -105.0, 'location': 'Wellington', 'country': 'United States of America (the)'},
    5: {'lat': 29.770031, 'long': -95.517364, 'location': 'Hedwig Village', 'country': 'United States of America (the)'},
    6: {'lat': 39.780622, 'long': -86.159535, 'location': 'Indianapolis', 'country': 'United States of America (the)'},
    7: {'lat': 38.961694, 'long': -96.596704, 'location': 'Ogden', 'country': 'United States of America (the)'},
    8: {'lat': 34.05, 'long': -118.25, 'location': 'Los Angeles', 'country': 'United States of America (the)'},
    9: {'lat': 40.7833, 'long': -73.9667, 'location': 'Manhattan', 'country': 'United States of America (the)'},
    10: {'lat': 37.38575, 'long': -122.02553, 'location': 'Sunnyvale', 'country': 'United States of America (the)'},
    11: {'lat': 47.6, 'long': -122.3, 'location': 'Seattle', 'country': 'United States of America (the)'},
    12: {'lat': 38.897303, 'long': -77.026842, 'location': 'Washington, D.C.', 'country': 'United States of America (the)'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 198.5472663728527},
    2.0: {'source': 2.0, 'destination': 5.0, 'fiber_length': 1500.0},
    3.0: {'source': 2.0, 'destination': 6.0, 'fiber_length': 885.1138413916553},
    4.0: {'source': 2.0, 'destination': 12.0, 'fiber_length': 1348.854429653284},
    5.0: {'source': 3.0, 'destination': 6.0, 'fiber_length': 388.6491313578985},
    6.0: {'source': 3.0, 'destination': 9.0, 'fiber_length': 1500.0},
    7.0: {'source': 4.0, 'destination': 7.0, 'fiber_length': 1116.022139449131},
    8.0: {'source': 4.0, 'destination': 10.0, 'fiber_length': 1892.498311036556},
    9.0: {'source': 4.0, 'destination': 11.0, 'fiber_length': 1963.718340246905},
    10.0: {'source': 5.0, 'destination': 7.0, 'fiber_length': 1500.0},
    11.0: {'source': 5.0, 'destination': 8.0, 'fiber_length': 2741.202050417932},
    12.0: {'source': 6.0, 'destination': 7.0, 'fiber_length': 1351.895189767641},
    13.0: {'source': 8.0, 'destination': 10.0, 'fiber_length': 755.4741126799757},
    14.0: {'source': 9.0, 'destination': 12.0, 'fiber_length': 502.4825029191594},
    15.0: {'source': 10.0, 'destination': 11.0, 'fiber_length': 1500.0},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
