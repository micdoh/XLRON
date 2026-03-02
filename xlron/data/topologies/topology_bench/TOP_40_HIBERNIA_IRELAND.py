def create_hibernia_ireland_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 53.34, 'long': -6.27, 'location': 'Rathmines', 'country': 'Ireland'},
    2: {'lat': 53.27, 'long': -9.05, 'location': 'Gaillimh', 'country': 'Ireland'},
    3: {'lat': 52.66, 'long': -8.62, 'location': 'Luimneach', 'country': 'Ireland'},
    4: {'lat': 51.9, 'long': -8.5, 'location': 'Cork', 'country': 'Ireland'},
    5: {'lat': 52.26, 'long': -7.11, 'location': 'Waterford', 'country': 'Ireland'},
    6: {'lat': 53.03, 'long': -7.3, 'location': 'Portlaoise', 'country': 'Ireland'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 5.0, 'fiber_length': 199.0550131750707},
    2.0: {'source': 1.0, 'destination': 6.0, 'fiber_length': 115.1998170443601},
    3.0: {'source': 2.0, 'destination': 3.0, 'fiber_length': 110.5333742112452},
    4.0: {'source': 3.0, 'destination': 4.0, 'fiber_length': 127.3522417022157},
    5.0: {'source': 3.0, 'destination': 6.0, 'fiber_length': 146.5940649422001},
    6.0: {'source': 4.0, 'destination': 5.0, 'fiber_length': 154.6128695737805},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
