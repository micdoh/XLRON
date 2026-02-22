def create_aarnet_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': -33.87, 'long': 151.21, 'location': 'Sydney', 'country': 'Australia'},
    2: {'lat': -27.47, 'long': 153.03, 'location': 'Brisbane', 'country': 'Australia'},
    3: {'lat': -35.28, 'long': 149.13, 'location': 'Canberra', 'country': 'Australia'},
    4: {'lat': -33.87, 'long': 151.21, 'location': 'Sydney', 'country': 'Australia'},
    5: {'lat': -19.25, 'long': 146.8, 'location': 'Townsville', 'country': 'Australia'},
    6: {'lat': -16.92, 'long': 145.77, 'location': 'Cairns', 'country': 'Australia'},
    7: {'lat': -27.47, 'long': 153.03, 'location': 'Brisbane', 'country': 'Australia'},
    8: {'lat': -23.38, 'long': 150.5, 'location': 'Rockhampton', 'country': 'Australia'},
    9: {'lat': -30.52, 'long': 151.65, 'location': 'Armidale', 'country': 'Australia'},
    10: {'lat': -42.88, 'long': 147.33, 'location': 'Hobart', 'country': 'Australia'},
    11: {'lat': -35.28, 'long': 149.13, 'location': 'Canberra', 'country': 'Australia'},
    12: {'lat': -31.93, 'long': 115.83, 'location': 'West Leederville', 'country': 'Australia'},
    13: {'lat': -31.93, 'long': 115.83, 'location': 'West Leederville', 'country': 'Australia'},
    14: {'lat': -34.93, 'long': 138.6, 'location': 'Adelaide', 'country': 'Australia'},
    15: {'lat': -34.93, 'long': 138.6, 'location': 'Adelaide', 'country': 'Australia'},
    16: {'lat': -37.81, 'long': 144.96, 'location': 'Melbourne', 'country': 'Australia'},
    17: {'lat': -37.81, 'long': 144.96, 'location': 'Melbourne', 'country': 'Australia'},
    18: {'lat': -23.7, 'long': 133.88, 'location': 'Alice Springs', 'country': 'Australia'},
    19: {'lat': -12.46, 'long': 130.84, 'location': 'Darwin', 'country': 'Australia'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 11.0, 'fiber_length': 369.9923322059562},
    2.0: {'source': 1.0, 'destination': 4.0, 'fiber_length': 0.0},
    3.0: {'source': 1.0, 'destination': 7.0, 'fiber_length': 1098.872498612245},
    4.0: {'source': 2.0, 'destination': 4.0, 'fiber_length': 1098.872498612245},
    5.0: {'source': 2.0, 'destination': 7.0, 'fiber_length': 0.0},
    6.0: {'source': 3.0, 'destination': 11.0, 'fiber_length': 0.0},
    7.0: {'source': 3.0, 'destination': 16.0, 'fiber_length': 700.0853166215753},
    8.0: {'source': 4.0, 'destination': 9.0, 'fiber_length': 562.1931537518077},
    9.0: {'source': 4.0, 'destination': 17.0, 'fiber_length': 1070.065123037869},
    10.0: {'source': 5.0, 'destination': 6.0, 'fiber_length': 421.5391557204086},
    11.0: {'source': 5.0, 'destination': 8.0, 'fiber_length': 897.1200628431984},
    12.0: {'source': 7.0, 'destination': 8.0, 'fiber_length': 781.3522199157314},
    13.0: {'source': 10.0, 'destination': 17.0, 'fiber_length': 897.5965341676058},
    14.0: {'source': 10.0, 'destination': 16.0, 'fiber_length': 897.5965341676058},
    15.0: {'source': 12.0, 'destination': 13.0, 'fiber_length': 0.0},
    16.0: {'source': 12.0, 'destination': 14.0, 'fiber_length': 2668.041349004836},
    17.0: {'source': 13.0, 'destination': 15.0, 'fiber_length': 2668.041349004836},
    18.0: {'source': 14.0, 'destination': 19.0, 'fiber_length': 3272.56429122365},
    19.0: {'source': 14.0, 'destination': 15.0, 'fiber_length': 0.0},
    20.0: {'source': 14.0, 'destination': 16.0, 'fiber_length': 979.627800989041},
    21.0: {'source': 15.0, 'destination': 17.0, 'fiber_length': 979.627800989041},
    22.0: {'source': 15.0, 'destination': 18.0, 'fiber_length': 1661.789527346283},
    23.0: {'source': 16.0, 'destination': 17.0, 'fiber_length': 0.0},
    24.0: {'source': 18.0, 'destination': 19.0, 'fiber_length': 1612.886647834685},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
