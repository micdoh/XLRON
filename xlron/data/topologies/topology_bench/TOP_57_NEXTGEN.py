def create_nextgen_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': -32.25, 'long': 148.62, 'location': 'Dubbo', 'country': 'Australia'},
    2: {'lat': -29.47, 'long': 149.85, 'location': 'Moree', 'country': 'Australia'},
    3: {'lat': -33.13, 'long': 148.18, 'location': 'Parkes', 'country': 'Australia'},
    4: {'lat': -36.08, 'long': 146.92, 'location': 'Albury', 'country': 'Australia'},
    5: {'lat': -32.93, 'long': 151.78, 'location': 'Newcastle', 'country': 'Australia'},
    6: {'lat': -30.3, 'long': 153.13, 'location': 'Coffs Harbour', 'country': 'Australia'},
    7: {'lat': -38.15, 'long': 144.36, 'location': 'Geelong', 'country': 'Australia'},
    8: {'lat': -36.38, 'long': 145.4, 'location': 'Shepparton', 'country': 'Australia'},
    9: {'lat': -31.93, 'long': 115.83, 'location': 'West Leederville', 'country': 'Australia'},
    10: {'lat': -34.93, 'long': 138.6, 'location': 'Adelaide', 'country': 'Australia'},
    11: {'lat': -37.81, 'long': 144.96, 'location': 'Melbourne', 'country': 'Australia'},
    12: {'lat': -33.87, 'long': 151.21, 'location': 'Sydney', 'country': 'Australia'},
    13: {'lat': -35.28, 'long': 149.13, 'location': 'Canberra', 'country': 'Australia'},
    14: {'lat': -27.47, 'long': 153.03, 'location': 'Brisbane', 'country': 'Australia'},
    15: {'lat': -30.75, 'long': 121.47, 'location': 'Kalgoorlie', 'country': 'Australia'},
    16: {'lat': -32.5, 'long': 137.77, 'location': 'Port Augusta', 'country': 'Australia'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 495.9873191580491},
    2.0: {'source': 1.0, 'destination': 3.0, 'fiber_length': 159.2426685232252},
    3.0: {'source': 2.0, 'destination': 14.0, 'fiber_length': 573.2539047554518},
    4.0: {'source': 4.0, 'destination': 11.0, 'fiber_length': 389.2413693493163},
    5.0: {'source': 4.0, 'destination': 13.0, 'fiber_length': 327.7931314787214},
    6.0: {'source': 5.0, 'destination': 12.0, 'fiber_length': 175.7293459970856},
    7.0: {'source': 5.0, 'destination': 6.0, 'fiber_length': 478.7282571610986},
    8.0: {'source': 6.0, 'destination': 14.0, 'fiber_length': 472.2482457284324},
    9.0: {'source': 7.0, 'destination': 10.0, 'fiber_length': 940.0318518372522},
    10.0: {'source': 7.0, 'destination': 11.0, 'fiber_length': 97.15069430135588},
    11.0: {'source': 8.0, 'destination': 11.0, 'fiber_length': 245.5904187693494},
    12.0: {'source': 9.0, 'destination': 10.0, 'fiber_length': 2668.041349004836},
    13.0: {'source': 9.0, 'destination': 15.0, 'fiber_length': 827.0961601404766},
    14.0: {'source': 10.0, 'destination': 16.0, 'fiber_length': 421.3409684441467},
    15.0: {'source': 12.0, 'destination': 13.0, 'fiber_length': 369.9923322059562},
    16.0: {'source': 15.0, 'destination': 16.0, 'fiber_length': 1942.481536905293},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
