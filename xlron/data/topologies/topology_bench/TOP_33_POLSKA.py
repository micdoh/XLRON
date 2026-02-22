def create_polska_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 54.2, 'long': 18.6, 'location': 'Legowo', 'country': 'Poland'},
    2: {'lat': 53.1, 'long': 17.9, 'location': 'Biale Blota', 'country': 'Poland'},
    3: {'lat': 54.2, 'long': 16.1, 'location': 'Mielno', 'country': 'Poland'},
    4: {'lat': 50.3, 'long': 18.8, 'location': 'Zabrze', 'country': 'Poland'},
    5: {'lat': 50.0, 'long': 19.8, 'location': 'Piekary', 'country': 'Poland'},
    6: {'lat': 53.1, 'long': 23.1, 'location': 'Bialystok', 'country': 'Poland'},
    7: {'lat': 51.7, 'long': 19.4, 'location': 'Ksawerow', 'country': 'Poland'},
    8: {'lat': 52.4, 'long': 16.8, 'location': 'Plewiska', 'country': 'Poland'},
    9: {'lat': 50.0, 'long': 21.9, 'location': 'Niechobrz', 'country': 'Poland'},
    10: {'lat': 53.4, 'long': 14.5, 'location': 'Przeclaw', 'country': 'Poland'},
    11: {'lat': 52.2, 'long': 21.0, 'location': 'Ochota', 'country': 'Poland'},
    12: {'lat': 51.1, 'long': 16.9, 'location': 'Smolec', 'country': 'Poland'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 11.0, 'fiber_length': 410.7744042567559},
    2.0: {'source': 1.0, 'destination': 3.0, 'fiber_length': 243.9034925586963},
    3.0: {'source': 1.0, 'destination': 6.0, 'fiber_length': 481.1108566271961},
    4.0: {'source': 2.0, 'destination': 3.0, 'fiber_length': 255.5770705699139},
    5.0: {'source': 2.0, 'destination': 8.0, 'fiber_length': 161.1319834866405},
    6.0: {'source': 2.0, 'destination': 11.0, 'fiber_length': 347.7170275587544},
    7.0: {'source': 3.0, 'destination': 10.0, 'fiber_length': 206.5017098295303},
    8.0: {'source': 4.0, 'destination': 5.0, 'fiber_length': 118.009331406665},
    9.0: {'source': 4.0, 'destination': 7.0, 'fiber_length': 241.8510106704026},
    10.0: {'source': 4.0, 'destination': 12.0, 'fiber_length': 241.0147228136016},
    11.0: {'source': 5.0, 'destination': 9.0, 'fiber_length': 225.1379758967965},
    12.0: {'source': 5.0, 'destination': 11.0, 'fiber_length': 387.8574867354514},
    13.0: {'source': 6.0, 'destination': 9.0, 'fiber_length': 531.804038576357},
    14.0: {'source': 6.0, 'destination': 11.0, 'fiber_length': 260.1554864729655},
    15.0: {'source': 7.0, 'destination': 11.0, 'fiber_length': 184.4117882015071},
    16.0: {'source': 7.0, 'destination': 12.0, 'fiber_length': 278.712944604},
    17.0: {'source': 8.0, 'destination': 10.0, 'fiber_length': 285.227139412663},
    18.0: {'source': 8.0, 'destination': 12.0, 'fiber_length': 217.0757821935017},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
