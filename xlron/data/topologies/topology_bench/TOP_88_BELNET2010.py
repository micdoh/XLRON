def create_belnet2010_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 50.67, 'long': 4.61, 'location': 'Louvain-la-Neuve', 'country': 'Belgium'},
    2: {'lat': 50.42, 'long': 4.43, 'location': 'Charleroi', 'country': 'Belgium'},
    3: {'lat': 49.68, 'long': 5.82, 'location': 'Arlon', 'country': 'Belgium'},
    4: {'lat': 50.47, 'long': 4.83, 'location': 'Namur', 'country': 'Belgium'},
    5: {'lat': 51.22, 'long': 4.42, 'location': 'Antwerpen', 'country': 'Belgium'},
    6: {'lat': 51.22, 'long': 4.42, 'location': 'Antwerpen', 'country': 'Belgium'},
    7: {'lat': 50.45, 'long': 3.95, 'location': 'Mons', 'country': 'Belgium'},
    8: {'lat': 50.83, 'long': 3.26, 'location': 'Kortrijk', 'country': 'Belgium'},
    9: {'lat': 51.05, 'long': 3.72, 'location': 'Gent', 'country': 'Belgium'},
    10: {'lat': 51.05, 'long': 3.72, 'location': 'Gent', 'country': 'Belgium'},
    11: {'lat': 50.53, 'long': 5.28, 'location': 'Amay', 'country': 'Belgium'},
    12: {'lat': 50.64, 'long': 5.57, 'location': 'Liege', 'country': 'Belgium'},
    13: {'lat': 51.21, 'long': 3.22, 'location': 'Brugge', 'country': 'Belgium'},
    14: {'lat': 51.17, 'long': 5.0, 'location': 'Geel', 'country': 'Belgium'},
    15: {'lat': 50.93, 'long': 5.33, 'location': 'Hasselt', 'country': 'Belgium'},
    16: {'lat': 50.88, 'long': 4.7, 'location': 'Leuven', 'country': 'Belgium'},
    17: {'lat': 50.88, 'long': 4.7, 'location': 'Leuven', 'country': 'Belgium'},
    18: {'lat': 50.92, 'long': 4.41, 'location': 'Vilvoorde', 'country': 'Belgium'},
    19: {'lat': 50.87, 'long': 4.4, 'location': 'Diegem', 'country': 'Belgium'},
    20: {'lat': 50.85, 'long': 4.35, 'location': 'Brussels', 'country': 'Belgium'},
    21: {'lat': 50.85, 'long': 4.35, 'location': 'Brussels', 'country': 'Belgium'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 45.85541115647035},
    2.0: {'source': 1.0, 'destination': 4.0, 'fiber_length': 40.69333334720884},
    3.0: {'source': 2.0, 'destination': 7.0, 'fiber_length': 51.2395076998644},
    4.0: {'source': 3.0, 'destination': 4.0, 'fiber_length': 169.0908680710074},
    5.0: {'source': 3.0, 'destination': 12.0, 'fiber_length': 162.3335148661989},
    6.0: {'source': 5.0, 'destination': 19.0, 'fiber_length': 58.41499742050551},
    7.0: {'source': 5.0, 'destination': 6.0, 'fiber_length': 0.0},
    8.0: {'source': 6.0, 'destination': 9.0, 'fiber_length': 78.55738294278487},
    9.0: {'source': 7.0, 'destination': 18.0, 'fiber_length': 92.24075796464625},
    10.0: {'source': 8.0, 'destination': 18.0, 'fiber_length': 121.9616993102718},
    11.0: {'source': 8.0, 'destination': 13.0, 'fiber_length': 63.51990117787189},
    12.0: {'source': 9.0, 'destination': 10.0, 'fiber_length': 0.0},
    13.0: {'source': 10.0, 'destination': 13.0, 'fiber_length': 58.74685908352896},
    14.0: {'source': 11.0, 'destination': 19.0, 'fiber_length': 108.8961988617577},
    15.0: {'source': 11.0, 'destination': 12.0, 'fiber_length': 35.77451614069067},
    16.0: {'source': 14.0, 'destination': 18.0, 'fiber_length': 74.6092837182215},
    17.0: {'source': 14.0, 'destination': 15.0, 'fiber_length': 52.9118055433793},
    18.0: {'source': 15.0, 'destination': 16.0, 'fiber_length': 66.78631425271848},
    19.0: {'source': 16.0, 'destination': 17.0, 'fiber_length': 0.0},
    20.0: {'source': 17.0, 'destination': 19.0, 'fiber_length': 31.61851948943619},
    21.0: {'source': 18.0, 'destination': 19.0, 'fiber_length': 8.405713815371765},
    22.0: {'source': 18.0, 'destination': 21.0, 'fiber_length': 13.27318355966539},
    23.0: {'source': 19.0, 'destination': 20.0, 'fiber_length': 6.232075587144267},
    24.0: {'source': 20.0, 'destination': 21.0, 'fiber_length': 0.0},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
