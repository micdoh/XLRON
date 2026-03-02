def create_telecomserbia_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 45.25, 'long': 19.84, 'location': 'Novi Sad', 'country': 'Serbia'},
    2: {'lat': 44.8, 'long': 20.47, 'location': 'Belgrade', 'country': 'Serbia'},
    3: {'lat': 44.02, 'long': 20.92, 'location': 'Kragujevac', 'country': 'Serbia'},
    4: {'lat': 43.32, 'long': 21.9, 'location': 'Nis', 'country': 'Serbia'},
    5: {'lat': 43.58, 'long': 21.33, 'location': 'Krusevac', 'country': 'Serbia'},
    6: {'lat': 42.44, 'long': 19.26, 'location': 'Podgorica', 'country': 'Montenegro'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 105.5904827819614},
    2.0: {'source': 1.0, 'destination': 6.0, 'fiber_length': 473.8473643351294},
    3.0: {'source': 2.0, 'destination': 3.0, 'fiber_length': 140.7127501064379},
    4.0: {'source': 3.0, 'destination': 4.0, 'fiber_length': 166.162096751931},
    5.0: {'source': 4.0, 'destination': 5.0, 'fiber_length': 81.5124068971401},
    6.0: {'source': 5.0, 'destination': 6.0, 'fiber_length': 316.0425249015269},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
