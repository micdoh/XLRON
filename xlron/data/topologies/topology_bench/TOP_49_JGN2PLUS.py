def create_jgn2plus_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 34.5, 'long': 133.0, 'location': 'Mihara', 'country': 'Japan'},
    2: {'lat': 35.0, 'long': 135.5, 'location': 'Kameoka', 'country': 'Japan'},
    3: {'lat': 33.0, 'long': 131.0, 'location': 'Aso', 'country': 'Japan'},
    4: {'lat': 33.75, 'long': 133.5, 'location': 'Kochi-shi', 'country': 'Japan'},
    5: {'lat': 43.66, 'long': 145.13, 'location': 'Shibetsu', 'country': 'Japan'},
    6: {'lat': 35.03, 'long': 136.9, 'location': 'Obu', 'country': 'Japan'},
    7: {'lat': 37.0, 'long': 137.5, 'location': 'Nyuzen', 'country': 'Japan'},
    8: {'lat': 26.34, 'long': 127.8, 'location': 'Okinawa', 'country': 'Japan'},
    9: {'lat': 43.06, 'long': 141.35, 'location': 'Sapporo', 'country': 'Japan'},
    10: {'lat': 38.27, 'long': 140.87, 'location': 'Sendai-shi', 'country': 'Japan'},
    11: {'lat': 35.61, 'long': 139.58, 'location': 'Chofugaoka', 'country': 'Japan'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 352.6038111726988},
    2.0: {'source': 1.0, 'destination': 3.0, 'fiber_length': 373.5148152658263},
    3.0: {'source': 2.0, 'destination': 4.0, 'fiber_length': 345.3459715788485},
    4.0: {'source': 2.0, 'destination': 6.0, 'fiber_length': 191.3084647985127},
    5.0: {'source': 3.0, 'destination': 8.0, 'fiber_length': 1203.552674046391},
    6.0: {'source': 5.0, 'destination': 11.0, 'fiber_length': 1500.0},
    7.0: {'source': 6.0, 'destination': 11.0, 'fiber_length': 377.323250985928},
    8.0: {'source': 7.0, 'destination': 11.0, 'fiber_length': 363.184659100073},
    9.0: {'source': 9.0, 'destination': 10.0, 'fiber_length': 801.2359389789707},
    10.0: {'source': 10.0, 'destination': 11.0, 'fiber_length': 475.8156762525535},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
