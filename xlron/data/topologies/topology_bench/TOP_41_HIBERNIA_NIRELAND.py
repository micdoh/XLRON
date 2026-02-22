def create_hibernia_nireland_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 53.72, 'long': -6.35, 'location': 'Drogheda', 'country': 'Ireland'},
    2: {'lat': 53.65, 'long': -3.01, 'location': 'Southport', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    3: {'lat': 54.41, 'long': -6.45, 'location': 'Portadown', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    4: {'lat': 54.12, 'long': -6.73, 'location': 'Castleblayney', 'country': 'Ireland'},
    5: {'lat': 54.25, 'long': -6.97, 'location': 'Monaghan', 'country': 'Ireland'},
    6: {'lat': 54.0, 'long': -6.42, 'location': 'Dundalk', 'country': 'Ireland'},
    7: {'lat': 53.34, 'long': -6.27, 'location': 'Rathmines', 'country': 'Ireland'},
    8: {'lat': 54.35, 'long': -6.67, 'location': 'Armagh', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    9: {'lat': 54.86, 'long': -6.28, 'location': 'Ballymena', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    10: {'lat': 54.58, 'long': -5.93, 'location': 'Belfast', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    11: {'lat': 54.6, 'long': -7.3, 'location': 'Omagh', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    12: {'lat': 54.82, 'long': -7.47, 'location': 'Strabane', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    13: {'lat': 54.95, 'long': -7.73, 'location': 'Letterkenny', 'country': 'Ireland'},
    14: {'lat': 55.0, 'long': -7.31, 'location': 'Derry', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    15: {'lat': 55.13, 'long': -6.67, 'location': 'Coleraine', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 6.0, 'fiber_length': 47.20674945464725},
    2.0: {'source': 1.0, 'destination': 7.0, 'fiber_length': 63.8754214064265},
    3.0: {'source': 2.0, 'destination': 7.0, 'fiber_length': 327.5450566295908},
    4.0: {'source': 3.0, 'destination': 10.0, 'fiber_length': 57.80376021398889},
    5.0: {'source': 3.0, 'destination': 8.0, 'fiber_length': 23.59811242401456},
    6.0: {'source': 4.0, 'destination': 5.0, 'fiber_length': 31.91952361094761},
    7.0: {'source': 4.0, 'destination': 6.0, 'fiber_length': 36.35382448093984},
    8.0: {'source': 5.0, 'destination': 8.0, 'fiber_length': 33.62707158090586},
    9.0: {'source': 8.0, 'destination': 11.0, 'fiber_length': 73.93671887572668},
    10.0: {'source': 9.0, 'destination': 10.0, 'fiber_length': 57.60115468007915},
    11.0: {'source': 9.0, 'destination': 15.0, 'fiber_length': 58.48471218608654},
    12.0: {'source': 11.0, 'destination': 12.0, 'fiber_length': 40.18465425949147},
    13.0: {'source': 12.0, 'destination': 14.0, 'fiber_length': 33.71513252579204},
    14.0: {'source': 12.0, 'destination': 13.0, 'fiber_length': 33.05150935102041},
    15.0: {'source': 13.0, 'destination': 14.0, 'fiber_length': 41.0614143104585},
    16.0: {'source': 14.0, 'destination': 15.0, 'fiber_length': 64.85982213364663},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
