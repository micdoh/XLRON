def create_funet_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 60.87, 'long': 26.7, 'location': 'Kouvola', 'country': 'Finland'},
    2: {'lat': 60.98, 'long': 25.66, 'location': 'Lahti', 'country': 'Finland'},
    3: {'lat': 61.69, 'long': 27.27, 'location': 'Mikkeli', 'country': 'Finland'},
    4: {'lat': 62.23, 'long': 25.73, 'location': 'Jyvaeskylae', 'country': 'Finland'},
    5: {'lat': 62.8, 'long': 22.83, 'location': 'Seinaejoki', 'country': 'Finland'},
    6: {'lat': 61.48, 'long': 21.78, 'location': 'Pori', 'country': 'Finland'},
    7: {'lat': 61.5, 'long': 23.79, 'location': 'Tampere', 'country': 'Finland'},
    8: {'lat': 61.0, 'long': 24.46, 'location': 'Haemeenlinna', 'country': 'Finland'},
    9: {'lat': 61.13, 'long': 21.51, 'location': 'Rauma', 'country': 'Finland'},
    10: {'lat': 60.45, 'long': 22.27, 'location': 'Turku', 'country': 'Finland'},
    11: {'lat': 60.47, 'long': 26.92, 'location': 'Kotka', 'country': 'Finland'},
    12: {'lat': 60.17, 'long': 24.94, 'location': 'Helsinki', 'country': 'Finland'},
    13: {'lat': 60.21, 'long': 24.65, 'location': 'Espoo', 'country': 'Finland'},
    14: {'lat': 60.38, 'long': 23.13, 'location': 'Salo', 'country': 'Finland'},
    15: {'lat': 62.6, 'long': 29.77, 'location': 'Joensuu', 'country': 'Finland'},
    16: {'lat': 61.06, 'long': 28.19, 'location': 'Lappeenranta', 'country': 'Finland'},
    17: {'lat': 65.01, 'long': 25.47, 'location': 'Oulu', 'country': 'Finland'},
    18: {'lat': 65.18, 'long': 25.35, 'location': 'Haukipudas', 'country': 'Finland'},
    19: {'lat': 66.5, 'long': 25.72, 'location': 'Rovaniemi', 'country': 'Finland'},
    20: {'lat': 67.42, 'long': 26.6, 'location': 'Sodankylae', 'country': 'Finland'},
    21: {'lat': 64.5, 'long': 28.22, 'location': 'Ristijaervi', 'country': 'Finland'},
    22: {'lat': 63.84, 'long': 23.13, 'location': 'Kokkola', 'country': 'Finland'},
    23: {'lat': 63.1, 'long': 21.62, 'location': 'Vaasa', 'country': 'Finland'},
    24: {'lat': 62.89, 'long': 27.68, 'location': 'Kuopio', 'country': 'Finland'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 86.26813317125257},
    2.0: {'source': 1.0, 'destination': 11.0, 'fiber_length': 69.09568740311218},
    3.0: {'source': 1.0, 'destination': 3.0, 'fiber_length': 144.1965983644955},
    4.0: {'source': 1.0, 'destination': 16.0, 'fiber_length': 124.7085175355978},
    5.0: {'source': 2.0, 'destination': 12.0, 'fiber_length': 147.4201259926803},
    6.0: {'source': 3.0, 'destination': 24.0, 'fiber_length': 202.6600285101637},
    7.0: {'source': 3.0, 'destination': 4.0, 'fiber_length': 150.6323530292557},
    8.0: {'source': 4.0, 'destination': 7.0, 'fiber_length': 195.195695505043},
    9.0: {'source': 5.0, 'destination': 23.0, 'fiber_length': 104.5322038934294},
    10.0: {'source': 5.0, 'destination': 6.0, 'fiber_length': 234.8775019654867},
    11.0: {'source': 6.0, 'destination': 9.0, 'fiber_length': 62.2530885907188},
    12.0: {'source': 7.0, 'destination': 8.0, 'fiber_length': 99.2162348130858},
    13.0: {'source': 8.0, 'destination': 13.0, 'fiber_length': 132.6807485603897},
    14.0: {'source': 9.0, 'destination': 10.0, 'fiber_length': 129.1903800084547},
    15.0: {'source': 10.0, 'destination': 14.0, 'fiber_length': 71.77452056803335},
    16.0: {'source': 12.0, 'destination': 13.0, 'fiber_length': 24.95422619798337},
    17.0: {'source': 13.0, 'destination': 14.0, 'fiber_length': 128.787008362161},
    18.0: {'source': 15.0, 'destination': 16.0, 'fiber_length': 285.3835358247076},
    19.0: {'source': 15.0, 'destination': 24.0, 'fiber_length': 166.7983713185157},
    20.0: {'source': 17.0, 'destination': 18.0, 'fiber_length': 29.58091876383459},
    21.0: {'source': 17.0, 'destination': 21.0, 'fiber_length': 213.2936479072919},
    22.0: {'source': 17.0, 'destination': 22.0, 'fiber_length': 257.7850438447481},
    23.0: {'source': 18.0, 'destination': 19.0, 'fiber_length': 221.6090728913921},
    24.0: {'source': 19.0, 'destination': 20.0, 'fiber_length': 163.8450341129229},
    25.0: {'source': 19.0, 'destination': 21.0, 'fiber_length': 375.6714421259598},
    26.0: {'source': 21.0, 'destination': 24.0, 'fiber_length': 271.4832163981253},
    27.0: {'source': 22.0, 'destination': 23.0, 'fiber_length': 166.9925127743007},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
