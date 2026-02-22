def create_bren_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 43.835571, 'long': 25.965654, 'location': 'Ruse', 'country': 'Bulgaria'},
    2: {'lat': 42.696491, 'long': 23.32601, 'location': 'Choumen', 'country': 'Bulgaria'},
    3: {'lat': 43.21405, 'long': 27.914734, 'location': 'Varna', 'country': 'Bulgaria'},
    4: {'lat': 42.504791, 'long': 27.462637, 'location': 'Burgas', 'country': 'Bulgaria'},
    5: {'lat': 43.0756647759984, 'long': 25.6181872309471, 'location': 'Trnovo', 'country': 'Bulgaria'},
    6: {'lat': 42.8838819691562, 'long': 25.3149737976083, 'location': 'Gabrovo', 'country': 'Bulgaria'},
    7: {'lat': 42.4272119485047, 'long': 25.632925398511, 'location': 'StaraZagora', 'country': 'Bulgaria'},
    8: {'lat': 43.25, 'long': 24.37, 'location': 'Pleven', 'country': 'Bulgaria'},
    9: {'lat': 42.700196738745, 'long': 23.3218840373449, 'location': 'Sofia', 'country': 'Bulgaria'},
    10: {'lat': 42.1355852008223, 'long': 24.7470533705177, 'location': 'Plovdiv', 'country': 'Bulgaria'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 372.6},
    2.0: {'source': 1.0, 'destination': 5.0, 'fiber_length': 133.545},
    3.0: {'source': 2.0, 'destination': 3.0, 'fiber_length': 566.7},
    4.0: {'source': 3.0, 'destination': 4.0, 'fiber_length': 130.575},
    5.0: {'source': 4.0, 'destination': 7.0, 'fiber_length': 225.45},
    6.0: {'source': 7.0, 'destination': 6.0, 'fiber_length': 85.57499999999999},
    7.0: {'source': 7.0, 'destination': 10.0, 'fiber_length': 119.655},
    8.0: {'source': 6.0, 'destination': 5.0, 'fiber_length': 48.915},
    9.0: {'source': 5.0, 'destination': 8.0, 'fiber_length': 135.795},
    10.0: {'source': 8.0, 'destination': 9.0, 'fiber_length': 197.1},
    11.0: {'source': 9.0, 'destination': 10.0, 'fiber_length': 199.2},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
