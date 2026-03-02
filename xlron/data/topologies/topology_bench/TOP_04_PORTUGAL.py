def create_portugal_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 37.74757445, 'long': -25.6726782661463, 'location': 'Ponta Delgada', 'country': 'Portugal'},
    2: {'lat': 32.6496497, 'long': -16.9086783, 'location': 'Funchal', 'country': 'Portugal'},
    3: {'lat': 37.0162727, 'long': -7.9351771, 'location': 'Faro', 'country': 'Portugal'},
    4: {'lat': 37.1375808, 'long': -8.5368426, 'location': 'Portimao', 'country': 'Portugal'},
    5: {'lat': 37.956549, 'long': -8.8689639, 'location': 'Sines', 'country': 'Portugal'},
    6: {'lat': 38.0154479, 'long': -7.8650368, 'location': 'Beja', 'country': 'Portugal'},
    7: {'lat': 38.5707742, 'long': -7.9092808, 'location': 'Evora', 'country': 'Portugal'},
    8: {'lat': 38.36584625, 'long': -8.53125903367014, 'location': 'Alcacer do Sal', 'country': 'Portugal'},
    9: {'lat': 38.5241783, 'long': -8.8932341, 'location': 'Setubal', 'country': 'Portugal'},
    10: {'lat': 38.8806123, 'long': -7.1637237, 'location': 'Elvas', 'country': 'Portugal'},
    11: {'lat': 38.7077507, 'long': -9.1365919, 'location': 'Lisboa', 'country': 'Portugal'},
    12: {'lat': 39.2363637, 'long': -8.6867081, 'location': 'Santarem', 'country': 'Portugal'},
    13: {'lat': 39.4071857, 'long': -9.1346004, 'location': 'Caldas da Rainha', 'country': 'Portugal'},
    14: {'lat': 39.7437902, 'long': -8.8071119, 'location': 'Leiria', 'country': 'Portugal'},
    15: {'lat': 39.2076447, 'long': -7.72151335401534, 'location': 'Portalegre', 'country': 'Portugal'},
    16: {'lat': 39.97675825, 'long': -7.4460599299667, 'location': 'Castelo Branco', 'country': 'Portugal'},
    17: {'lat': 40.2111931, 'long': -8.4294632, 'location': 'Coimbra', 'country': 'Portugal'},
    18: {'lat': 40.7046066, 'long': -7.1951392360713, 'location': 'Guarda', 'country': 'Portugal'},
    19: {'lat': 40.6574713, 'long': -7.9138664, 'location': 'Viseu', 'country': 'Portugal'},
    20: {'lat': 40.640496, 'long': -8.6537841, 'location': 'Aveiro', 'country': 'Portugal'},
    21: {'lat': 41.1494512, 'long': -8.6107884, 'location': 'Porto', 'country': 'Portugal'},
    22: {'lat': 40.9020813, 'long': -8.4896358, 'location': 'Sao Joao da Madeira', 'country': 'Portugal'},
    23: {'lat': 41.5229299, 'long': -7.54663124811733, 'location': 'Vila Real', 'country': 'Portugal'},
    24: {'lat': 41.5510583, 'long': -8.4280045, 'location': 'Braga', 'country': 'Portugal'},
    25: {'lat': 41.8803257, 'long': -8.52325529355434, 'location': 'Viana do Castelo', 'country': 'Portugal'},
    26: {'lat': 41.5084468, 'long': -6.77330236053306, 'location': 'Braganca', 'country': 'Portugal'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 11.0, 'fiber_length': 1446.39945438625},
    2.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 976.742289572545},
    3.0: {'source': 2.0, 'destination': 11.0, 'fiber_length': 972.194188199617},
    4.0: {'source': 2.0, 'destination': 4.0, 'fiber_length': 911.57201807488},
    5.0: {'source': 3.0, 'destination': 4.0, 'fiber_length': 55.0543120685811},
    6.0: {'source': 3.0, 'destination': 6.0, 'fiber_length': 111.275446867203},
    7.0: {'source': 4.0, 'destination': 5.0, 'fiber_length': 95.6565180797956},
    8.0: {'source': 5.0, 'destination': 8.0, 'fiber_length': 54.2500947602716},
    9.0: {'source': 6.0, 'destination': 7.0, 'fiber_length': 61.8701551028065},
    10.0: {'source': 7.0, 'destination': 10.0, 'fiber_length': 73.2798208384217},
    11.0: {'source': 7.0, 'destination': 8.0, 'fiber_length': 58.7487592977907},
    12.0: {'source': 8.0, 'destination': 9.0, 'fiber_length': 36.1069913258553},
    13.0: {'source': 9.0, 'destination': 11.0, 'fiber_length': 29.3888654145231},
    14.0: {'source': 10.0, 'destination': 15.0, 'fiber_length': 60.3556832946721},
    15.0: {'source': 11.0, 'destination': 12.0, 'fiber_length': 70.4807828758445},
    16.0: {'source': 11.0, 'destination': 13.0, 'fiber_length': 77.7739210296261},
    17.0: {'source': 12.0, 'destination': 15.0, 'fiber_length': 83.2056964419818},
    18.0: {'source': 12.0, 'destination': 17.0, 'fiber_length': 110.606210171264},
    19.0: {'source': 12.0, 'destination': 14.0, 'fiber_length': 57.3615283020467},
    20.0: {'source': 13.0, 'destination': 14.0, 'fiber_length': 46.783879841642},
    21.0: {'source': 14.0, 'destination': 20.0, 'fiber_length': 100.556177793881},
    22.0: {'source': 15.0, 'destination': 16.0, 'fiber_length': 88.7187444306888},
    23.0: {'source': 16.0, 'destination': 18.0, 'fiber_length': 83.6804651678564},
    24.0: {'source': 17.0, 'destination': 19.0, 'fiber_length': 66.0816469231036},
    25.0: {'source': 17.0, 'destination': 22.0, 'fiber_length': 76.9913694518877},
    26.0: {'source': 17.0, 'destination': 20.0, 'fiber_length': 51.3741314750661},
    27.0: {'source': 18.0, 'destination': 19.0, 'fiber_length': 60.8325566979028},
    28.0: {'source': 19.0, 'destination': 23.0, 'fiber_length': 101.035816727768},
    29.0: {'source': 20.0, 'destination': 21.0, 'fiber_length': 56.7085842045847},
    30.0: {'source': 21.0, 'destination': 22.0, 'fiber_length': 29.3238210832733},
    31.0: {'source': 21.0, 'destination': 24.0, 'fiber_length': 47.1912026200565},
    32.0: {'source': 21.0, 'destination': 25.0, 'fiber_length': 81.5957703867763},
    33.0: {'source': 22.0, 'destination': 24.0, 'fiber_length': 72.3468822785827},
    34.0: {'source': 23.0, 'destination': 26.0, 'fiber_length': 64.4072956232962},
    35.0: {'source': 24.0, 'destination': 26.0, 'fiber_length': 137.819940126367},
    36.0: {'source': 24.0, 'destination': 25.0, 'fiber_length': 37.4567782591721},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
