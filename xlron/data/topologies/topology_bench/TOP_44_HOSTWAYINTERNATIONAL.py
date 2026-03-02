def create_hostwayinternational_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 47.61, 'long': -122.33, 'location': 'Seattle', 'country': 'United States of America (the)'},
    2: {'lat': 27.95, 'long': -82.46, 'location': 'Tampa', 'country': 'United States of America (the)'},
    3: {'lat': 41.85, 'long': -87.65, 'location': 'Chicago', 'country': 'United States of America (the)'},
    4: {'lat': 49.25, 'long': -123.12, 'location': 'Vancouver', 'country': 'Canada'},
    5: {'lat': 25.77, 'long': -80.19, 'location': 'Miami', 'country': 'United States of America (the)'},
    6: {'lat': 30.27, 'long': -97.74, 'location': 'Austin', 'country': 'United States of America (the)'},
    7: {'lat': 40.71, 'long': -74.01, 'location': 'New York City', 'country': 'United States of America (the)'},
    8: {'lat': 33.75, 'long': -84.39, 'location': 'Atlanta', 'country': 'United States of America (the)'},
    9: {'lat': 44.43, 'long': 26.11, 'location': 'Bucharest', 'country': 'Romania'},
    10: {'lat': 37.57, 'long': 126.98, 'location': 'Seoul', 'country': 'Korea (the Republic of)'},
    11: {'lat': -33.87, 'long': 151.21, 'location': 'Sydney', 'country': 'Australia'},
    12: {'lat': 52.37, 'long': 9.73, 'location': 'Hannover', 'country': 'Germany'},
    13: {'lat': 52.37, 'long': 4.89, 'location': 'Amsterdam', 'country': 'Netherlands (Kingdom of the)'},
    14: {'lat': 51.22, 'long': 4.42, 'location': 'Antwerpen', 'country': 'Belgium'},
    15: {'lat': 50.12, 'long': 8.68, 'location': 'Frankfurt am Main', 'country': 'Germany'},
    16: {'lat': 51.51, 'long': -0.13, 'location': 'London', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 3.0, 'fiber_length': 3485.643998403856},
    2.0: {'source': 1.0, 'destination': 4.0, 'fiber_length': 287.1681018437656},
    3.0: {'source': 1.0, 'destination': 6.0, 'fiber_length': 3559.806322172083},
    4.0: {'source': 2.0, 'destination': 5.0, 'fiber_length': 496.2580235875224},
    5.0: {'source': 2.0, 'destination': 8.0, 'fiber_length': 1006.029022461507},
    6.0: {'source': 3.0, 'destination': 6.0, 'fiber_length': 1965.575314978632},
    7.0: {'source': 3.0, 'destination': 8.0, 'fiber_length': 1417.377937950516},
    8.0: {'source': 5.0, 'destination': 11.0, 'fiber_length': 18783.10176773076},
    9.0: {'source': 5.0, 'destination': 7.0, 'fiber_length': 2195.769639674104},
    10.0: {'source': 5.0, 'destination': 8.0, 'fiber_length': 1462.933348688248},
    11.0: {'source': 7.0, 'destination': 11.0, 'fiber_length': 19985.49230622196},
    12.0: {'source': 7.0, 'destination': 8.0, 'fiber_length': 1500.0},
    13.0: {'source': 7.0, 'destination': 16.0, 'fiber_length': 6963.055714635429},
    14.0: {'source': 9.0, 'destination': 10.0, 'fiber_length': 9914.597335339473},
    15.0: {'source': 9.0, 'destination': 15.0, 'fiber_length': 1818.246624112978},
    16.0: {'source': 10.0, 'destination': 11.0, 'fiber_length': 10412.13123737522},
    17.0: {'source': 12.0, 'destination': 13.0, 'fiber_length': 492.7979258364858},
    18.0: {'source': 12.0, 'destination': 15.0, 'fiber_length': 390.956886368611},
    19.0: {'source': 13.0, 'destination': 14.0, 'fiber_length': 197.8428042192446},
    20.0: {'source': 13.0, 'destination': 16.0, 'fiber_length': 535.6163051270169},
    21.0: {'source': 14.0, 'destination': 15.0, 'fiber_length': 486.1756023156619},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
