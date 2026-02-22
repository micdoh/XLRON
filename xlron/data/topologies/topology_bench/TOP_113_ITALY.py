def create_italy_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 46.65594545, 'long': 11.23021287087526, 'location': 'Avelengo', 'country': 'Italy'},
    2: {'lat': 45.4641943, 'long': 9.1896346, 'location': 'Milano', 'country': 'Italy'},
    3: {'lat': 45.0677551, 'long': 7.6824892, 'location': 'Turin', 'country': 'Italy'},
    4: {'lat': 45.4384958, 'long': 10.9924122, 'location': 'Verona', 'country': 'Italy'},
    5: {'lat': 45.4371908, 'long': 12.3345898, 'location': 'Venice', 'country': 'Italy'},
    6: {'lat': 45.6496485, 'long': 13.7772781, 'location': 'Trieste', 'country': 'Italy'},
    7: {'lat': 44.4938203, 'long': 11.3426327, 'location': 'Bologna', 'country': 'Italy'},
    8: {'lat': 44.40726, 'long': 8.9338624, 'location': 'San Teodoro', 'country': 'Italy'},
    9: {'lat': 43.7159395, 'long': 10.4018624, 'location': 'Pisa', 'country': 'Italy'},
    10: {'lat': 43.7697955, 'long': 11.2556404, 'location': 'Florence', 'country': 'Italy'},
    11: {'lat': 43.1070321, 'long': 12.40299620990649, 'location': 'Perugia', 'country': 'Italy'},
    12: {'lat': 43.4801204, 'long': 13.21879060915176, 'location': 'Jesi', 'country': 'Italy'},
    13: {'lat': 41.8933203, 'long': 12.4829321, 'location': 'Rome', 'country': 'Italy'},
    14: {'lat': 42.3102619, 'long': 13.95759010584892, 'location': 'Alanno', 'country': 'Italy'},
    15: {'lat': 39.2171994, 'long': 9.113311, 'location': 'Cagliari', 'country': 'Italy'},
    16: {'lat': 40.8358846, 'long': 14.2487679, 'location': 'Napoli', 'country': 'Italy'},
    17: {'lat': 41.1257843, 'long': 16.8620293, 'location': 'Bari', 'country': 'Italy'},
    18: {'lat': 40.51731195, 'long': 15.82160882487105, 'location': 'Abriola', 'country': 'Italy'},
    19: {'lat': 38.82996034999999, 'long': 16.43155687627833, 'location': 'Girifalco', 'country': 'Italy'},
    20: {'lat': 38.1112268, 'long': 13.3524434, 'location': 'Palermo', 'country': 'Italy'},
    21: {'lat': 37.5023612, 'long': 15.0873718, 'location': 'Catania', 'country': 'Italy'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 308.6701335578152},
    2.0: {'source': 2.0, 'destination': 3.0, 'fiber_length': 188.8744179910006},
    3.0: {'source': 3.0, 'destination': 4.0, 'fiber_length': 393.5030002207023},
    4.0: {'source': 4.0, 'destination': 5.0, 'fiber_length': 157.0805125823537},
    5.0: {'source': 5.0, 'destination': 6.0, 'fiber_length': 172.2119468511727},
    6.0: {'source': 2.0, 'destination': 7.0, 'fiber_length': 301.1822884419588},
    7.0: {'source': 3.0, 'destination': 8.0, 'fiber_length': 184.7066745030278},
    8.0: {'source': 8.0, 'destination': 9.0, 'fiber_length': 210.3598030639492},
    9.0: {'source': 9.0, 'destination': 10.0, 'fiber_length': 103.2704857925649},
    10.0: {'source': 10.0, 'destination': 11.0, 'fiber_length': 177.5606364517409},
    11.0: {'source': 11.0, 'destination': 12.0, 'fiber_length': 116.9638164912841},
    12.0: {'source': 12.0, 'destination': 13.0, 'fiber_length': 279.6173118032349},
    13.0: {'source': 13.0, 'destination': 14.0, 'fiber_length': 195.2904697940523},
    14.0: {'source': 13.0, 'destination': 15.0, 'fiber_length': 617.6156991317341},
    15.0: {'source': 15.0, 'destination': 16.0, 'fiber_length': 709.1498404948832},
    16.0: {'source': 15.0, 'destination': 17.0, 'fiber_length': 1037.170533225839},
    17.0: {'source': 17.0, 'destination': 18.0, 'fiber_length': 165.9657761952489},
    18.0: {'source': 15.0, 'destination': 19.0, 'fiber_length': 950.226176616599},
    19.0: {'source': 19.0, 'destination': 20.0, 'fiber_length': 419.5549852305203},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
