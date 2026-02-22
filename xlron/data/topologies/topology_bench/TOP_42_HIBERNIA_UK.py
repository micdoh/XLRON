def create_hibernia_uk_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 51.51, 'long': -0.13, 'location': 'London', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    2: {'lat': 53.65, 'long': -3.01, 'location': 'Southport', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    3: {'lat': 53.48, 'long': -2.24, 'location': 'Manchester', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    4: {'lat': 52.57, 'long': -0.25, 'location': 'Peterborough', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    5: {'lat': 52.2, 'long': 0.12, 'location': 'Cambridge', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    6: {'lat': 53.38, 'long': -1.47, 'location': 'Sheffield', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    7: {'lat': 52.63, 'long': -1.13, 'location': 'Leicester', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    8: {'lat': 53.93, 'long': -2.21, 'location': 'Barnoldswick', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    9: {'lat': 53.8, 'long': -1.55, 'location': 'Leeds', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    10: {'lat': 52.47, 'long': -1.92, 'location': 'Birmingham', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    11: {'lat': 53.41, 'long': -2.98, 'location': 'Liverpool', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    12: {'lat': 51.46, 'long': -0.97, 'location': 'Reading', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    13: {'lat': 51.45, 'long': -2.58, 'location': 'Bristol', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 12.0, 'fiber_length': 87.64367488816515},
    2.0: {'source': 1.0, 'destination': 5.0, 'fiber_length': 117.9331652049437},
    3.0: {'source': 2.0, 'destination': 8.0, 'fiber_length': 91.62093159452056},
    4.0: {'source': 2.0, 'destination': 11.0, 'fiber_length': 40.14051422646284},
    5.0: {'source': 3.0, 'destination': 10.0, 'fiber_length': 171.4982865912685},
    6.0: {'source': 3.0, 'destination': 11.0, 'fiber_length': 74.43306683453827},
    7.0: {'source': 4.0, 'destination': 7.0, 'fiber_length': 89.7083602174358},
    8.0: {'source': 4.0, 'destination': 5.0, 'fiber_length': 72.29987564362703},
    9.0: {'source': 6.0, 'destination': 7.0, 'fiber_length': 129.6648190008794},
    10.0: {'source': 6.0, 'destination': 9.0, 'fiber_length': 70.49908780911265},
    11.0: {'source': 8.0, 'destination': 9.0, 'fiber_length': 68.44007076148796},
    12.0: {'source': 10.0, 'destination': 13.0, 'fiber_length': 183.1512309474041},
    13.0: {'source': 12.0, 'destination': 13.0, 'fiber_length': 167.3373835976718},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
