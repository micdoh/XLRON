def create_renater1999_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 44.83, 'long': -0.57, 'location': 'Bordeaux', 'country': 'France'},
    2: {'lat': 43.6, 'long': 1.44, 'location': 'Toulouse', 'country': 'France'},
    3: {'lat': 47.92, 'long': 1.9, 'location': 'Orleans', 'country': 'France'},
    4: {'lat': 45.83, 'long': 1.26, 'location': 'Limoges', 'country': 'France'},
    5: {'lat': 47.32, 'long': 5.02, 'location': 'Dijon', 'country': 'France'},
    6: {'lat': 47.25, 'long': 6.03, 'location': 'Besancon', 'country': 'France'},
    7: {'lat': 43.6, 'long': 3.88, 'location': 'Montpellier', 'country': 'France'},
    8: {'lat': 45.78, 'long': 3.08, 'location': 'Clermont-Ferrand', 'country': 'France'},
    9: {'lat': 45.17, 'long': 5.72, 'location': 'Grenoble', 'country': 'France'},
    10: {'lat': 43.7, 'long': 7.27, 'location': 'Nice', 'country': 'France'},
    11: {'lat': 45.75, 'long': 4.85, 'location': 'Lyon', 'country': 'France'},
    12: {'lat': 48.85, 'long': 2.35, 'location': 'Paris', 'country': 'France'},
    13: {'lat': 42.3, 'long': 9.15, 'location': 'Corte', 'country': 'France'},
    14: {'lat': 43.3, 'long': 5.4, 'location': 'Marseille 04', 'country': 'France'},
    15: {'lat': 47.22, 'long': -1.55, 'location': 'Nantes', 'country': 'France'},
    16: {'lat': 46.58, 'long': 0.33, 'location': 'Poitiers', 'country': 'France'},
    17: {'lat': 48.58, 'long': 7.75, 'location': 'Strasbourg', 'country': 'France'},
    18: {'lat': 48.68, 'long': 6.2, 'location': 'Jarville-la-Malgrange', 'country': 'France'},
    19: {'lat': 49.25, 'long': 4.03, 'location': 'Reims', 'country': 'France'},
    20: {'lat': 49.42, 'long': 2.83, 'location': 'Compiegne', 'country': 'France'},
    21: {'lat': 50.63, 'long': 3.07, 'location': 'Lille', 'country': 'France'},
    22: {'lat': 49.44, 'long': 1.1, 'location': 'Rouen', 'country': 'France'},
    23: {'lat': 49.19, 'long': -0.36, 'location': 'Caen', 'country': 'France'},
    24: {'lat': 48.08, 'long': -1.68, 'location': 'Rennes', 'country': 'France'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 16.0, 'fiber_length': 310.1353076025006},
    2.0: {'source': 2.0, 'destination': 12.0, 'fiber_length': 881.9148496472068},
    3.0: {'source': 2.0, 'destination': 7.0, 'fiber_length': 294.7081152484741},
    4.0: {'source': 3.0, 'destination': 12.0, 'fiber_length': 162.9284693466608},
    5.0: {'source': 3.0, 'destination': 16.0, 'fiber_length': 285.5552568765627},
    6.0: {'source': 4.0, 'destination': 16.0, 'fiber_length': 164.8408021406742},
    7.0: {'source': 5.0, 'destination': 12.0, 'fiber_length': 391.9101437019439},
    8.0: {'source': 5.0, 'destination': 6.0, 'fiber_length': 114.8694624115643},
    9.0: {'source': 8.0, 'destination': 11.0, 'fiber_length': 206.0046605848057},
    10.0: {'source': 9.0, 'destination': 11.0, 'fiber_length': 140.4186699208927},
    11.0: {'source': 10.0, 'destination': 13.0, 'fiber_length': 327.2726517839708},
    12.0: {'source': 10.0, 'destination': 14.0, 'fiber_length': 235.8713173227889},
    13.0: {'source': 11.0, 'destination': 12.0, 'fiber_length': 589.2508599924295},
    14.0: {'source': 11.0, 'destination': 14.0, 'fiber_length': 413.8389675135418},
    15.0: {'source': 12.0, 'destination': 18.0, 'fiber_length': 424.175933883086},
    16.0: {'source': 12.0, 'destination': 19.0, 'fiber_length': 195.388303552465},
    17.0: {'source': 12.0, 'destination': 20.0, 'fiber_length': 108.5465409911054},
    18.0: {'source': 12.0, 'destination': 21.0, 'fiber_length': 306.8625058255422},
    19.0: {'source': 12.0, 'destination': 22.0, 'fiber_length': 168.1761860997829},
    20.0: {'source': 12.0, 'destination': 24.0, 'fiber_length': 463.7727051434647},
    21.0: {'source': 15.0, 'destination': 24.0, 'fiber_length': 144.1831991731319},
    22.0: {'source': 17.0, 'destination': 18.0, 'fiber_length': 171.6753296158936},
    23.0: {'source': 22.0, 'destination': 23.0, 'fiber_length': 164.130662797804},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
