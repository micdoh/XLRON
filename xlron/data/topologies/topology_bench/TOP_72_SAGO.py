def create_sago_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 27.45, 'long': -80.33, 'location': 'Fort Pierce', 'country': 'United States of America (the)'},
    2: {'lat': 28.08, 'long': -80.61, 'location': 'Melbourne', 'country': 'United States of America (the)'},
    3: {'lat': 28.09, 'long': -81.72, 'location': 'Lake Alfred', 'country': 'United States of America (the)'},
    4: {'lat': 28.54, 'long': -81.38, 'location': 'Orlando', 'country': 'United States of America (the)'},
    5: {'lat': 26.72, 'long': -80.05, 'location': 'West Palm Beach', 'country': 'United States of America (the)'},
    6: {'lat': 27.95, 'long': -82.46, 'location': 'Tampa', 'country': 'United States of America (the)'},
    7: {'lat': 25.77, 'long': -80.19, 'location': 'Miami', 'country': 'United States of America (the)'},
    8: {'lat': 26.12, 'long': -80.14, 'location': 'Fort Lauderdale', 'country': 'United States of America (the)'},
    9: {'lat': 33.25, 'long': -83.9, 'location': 'Indian Springs', 'country': 'United States of America (the)'},
    10: {'lat': 33.75, 'long': -84.39, 'location': 'Atlanta', 'country': 'United States of America (the)'},
    11: {'lat': 32.32, 'long': -82.56, 'location': 'Soperton', 'country': 'United States of America (the)'},
    12: {'lat': 32.88, 'long': -83.33, 'location': 'Gordon', 'country': 'United States of America (the)'},
    13: {'lat': 30.94, 'long': -82.02, 'location': 'Folkston', 'country': 'United States of America (the)'},
    14: {'lat': 31.72, 'long': -82.2, 'location': 'Baxley', 'country': 'United States of America (the)'},
    15: {'lat': 29.89, 'long': -81.31, 'location': 'Saint Augustine', 'country': 'United States of America (the)'},
    16: {'lat': 30.33, 'long': -81.66, 'location': 'Jacksonville', 'country': 'United States of America (the)'},
    17: {'lat': 28.61, 'long': -80.81, 'location': 'Titusville', 'country': 'United States of America (the)'},
    18: {'lat': 29.21, 'long': -81.02, 'location': 'Daytona Beach', 'country': 'United States of America (the)'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 112.9130335236984},
    2.0: {'source': 1.0, 'destination': 5.0, 'fiber_length': 128.6622981191926},
    3.0: {'source': 2.0, 'destination': 17.0, 'fiber_length': 93.14768168607968},
    4.0: {'source': 3.0, 'destination': 4.0, 'fiber_length': 90.14377938900955},
    5.0: {'source': 3.0, 'destination': 6.0, 'fiber_length': 111.4326474233612},
    6.0: {'source': 4.0, 'destination': 18.0, 'fiber_length': 123.502526239912},
    7.0: {'source': 5.0, 'destination': 8.0, 'fiber_length': 100.9743372348429},
    8.0: {'source': 7.0, 'destination': 8.0, 'fiber_length': 58.85702770663077},
    9.0: {'source': 9.0, 'destination': 10.0, 'fiber_length': 107.7012067809803},
    10.0: {'source': 9.0, 'destination': 12.0, 'fiber_length': 100.7797542631112},
    11.0: {'source': 11.0, 'destination': 12.0, 'fiber_length': 142.935071154233},
    12.0: {'source': 11.0, 'destination': 14.0, 'fiber_length': 112.280376164828},
    13.0: {'source': 13.0, 'destination': 14.0, 'fiber_length': 132.6014600386424},
    14.0: {'source': 13.0, 'destination': 16.0, 'fiber_length': 114.1091942451913},
    15.0: {'source': 15.0, 'destination': 18.0, 'fiber_length': 120.972544952087},
    16.0: {'source': 15.0, 'destination': 16.0, 'fiber_length': 89.08495741337417},
    17.0: {'source': 17.0, 'destination': 18.0, 'fiber_length': 104.6670832176189},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
