def create_renater2010_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 44.83, 'long': -0.57, 'location': 'Bordeaux', 'country': 'France'},
    2: {'lat': 43.3, 'long': -0.37, 'location': 'Pau', 'country': 'France'},
    3: {'lat': 47.92, 'long': 1.9, 'location': 'Orleans', 'country': 'France'},
    4: {'lat': 45.83, 'long': 1.26, 'location': 'Limoges', 'country': 'France'},
    5: {'lat': 45.78, 'long': 3.08, 'location': 'Clermont-Ferrand', 'country': 'France'},
    6: {'lat': 47.32, 'long': 5.02, 'location': 'Dijon', 'country': 'France'},
    7: {'lat': 43.6, 'long': 1.44, 'location': 'Toulouse', 'country': 'France'},
    8: {'lat': 43.6, 'long': 3.88, 'location': 'Montpellier', 'country': 'France'},
    9: {'lat': 47.25, 'long': 6.03, 'location': 'Besancon', 'country': 'France'},
    10: {'lat': 45.17, 'long': 5.72, 'location': 'Grenoble', 'country': 'France'},
    11: {'lat': 48.73, 'long': -3.47, 'location': 'Lannion', 'country': 'France'},
    12: {'lat': 48.4, 'long': -4.48, 'location': 'Brest', 'country': 'France'},
    13: {'lat': 48.0, 'long': 0.2, 'location': 'Le Mans', 'country': 'France'},
    14: {'lat': 47.38, 'long': 0.68, 'location': 'Tours', 'country': 'France'},
    15: {'lat': 47.22, 'long': 2.08, 'location': 'Vierzon', 'country': 'France'},
    16: {'lat': 46.44, 'long': 6.47, 'location': 'Saint-Prex', 'country': 'Switzerland'},
    17: {'lat': 47.67, 'long': -2.75, 'location': 'Vannes', 'country': 'France'},
    18: {'lat': 47.75, 'long': -3.37, 'location': 'Lorient', 'country': 'France'},
    19: {'lat': 48.0, 'long': -4.1, 'location': 'Quimper', 'country': 'France'},
    20: {'lat': 48.52, 'long': -2.78, 'location': 'Saint-Brieuc', 'country': 'France'},
    21: {'lat': 47.47, 'long': -0.55, 'location': 'Angers', 'country': 'France'},
    22: {'lat': 42.3, 'long': 9.15, 'location': 'Corte', 'country': 'France'},
    23: {'lat': 43.7, 'long': 5.75, 'location': 'Vinon-sur-Verdon', 'country': 'France'},
    24: {'lat': 43.3, 'long': 5.4, 'location': 'Marseille 04', 'country': 'France'},
    25: {'lat': 43.7, 'long': 7.27, 'location': 'Nice', 'country': 'France'},
    26: {'lat': 45.75, 'long': 4.85, 'location': 'Lyon', 'country': 'France'},
    27: {'lat': 48.85, 'long': 2.35, 'location': 'Paris', 'country': 'France'},
    28: {'lat': 47.22, 'long': -1.55, 'location': 'Nantes', 'country': 'France'},
    29: {'lat': 46.58, 'long': 0.33, 'location': 'Poitiers', 'country': 'France'},
    30: {'lat': 48.58, 'long': 7.75, 'location': 'Strasbourg', 'country': 'France'},
    31: {'lat': 48.68, 'long': 6.2, 'location': 'Jarville-la-Malgrange', 'country': 'France'},
    32: {'lat': 49.25, 'long': 4.03, 'location': 'Reims', 'country': 'France'},
    33: {'lat': 49.42, 'long': 2.83, 'location': 'Compiegne', 'country': 'France'},
    34: {'lat': 50.63, 'long': 3.07, 'location': 'Lille', 'country': 'France'},
    35: {'lat': 49.44, 'long': 1.1, 'location': 'Rouen', 'country': 'France'},
    36: {'lat': 49.19, 'long': -0.36, 'location': 'Caen', 'country': 'France'},
    37: {'lat': 48.08, 'long': -1.68, 'location': 'Rennes', 'country': 'France'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 28.0, 'fiber_length': 414.4659365231481},
    2.0: {'source': 1.0, 'destination': 29.0, 'fiber_length': 310.1353076025006},
    3.0: {'source': 1.0, 'destination': 5.0, 'fiber_length': 456.5084316959787},
    4.0: {'source': 1.0, 'destination': 7.0, 'fiber_length': 315.9334161054513},
    5.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 256.3153501283776},
    6.0: {'source': 2.0, 'destination': 7.0, 'fiber_length': 224.8019465794579},
    7.0: {'source': 3.0, 'destination': 27.0, 'fiber_length': 162.9284693466608},
    8.0: {'source': 3.0, 'destination': 29.0, 'fiber_length': 285.5552568765627},
    9.0: {'source': 3.0, 'destination': 14.0, 'fiber_length': 164.018993814066},
    10.0: {'source': 3.0, 'destination': 15.0, 'fiber_length': 118.4986625531541},
    11.0: {'source': 4.0, 'destination': 29.0, 'fiber_length': 164.8408021406742},
    12.0: {'source': 4.0, 'destination': 5.0, 'fiber_length': 211.7736137604195},
    13.0: {'source': 5.0, 'destination': 26.0, 'fiber_length': 206.0046605848057},
    14.0: {'source': 6.0, 'destination': 9.0, 'fiber_length': 114.8694624115643},
    15.0: {'source': 6.0, 'destination': 26.0, 'fiber_length': 262.589306462549},
    16.0: {'source': 7.0, 'destination': 8.0, 'fiber_length': 294.7081152484741},
    17.0: {'source': 8.0, 'destination': 24.0, 'fiber_length': 190.7296902972907},
    18.0: {'source': 9.0, 'destination': 30.0, 'fiber_length': 293.5504767383885},
    19.0: {'source': 10.0, 'destination': 23.0, 'fiber_length': 245.2108392352735},
    20.0: {'source': 10.0, 'destination': 16.0, 'fiber_length': 229.0709496277423},
    21.0: {'source': 10.0, 'destination': 26.0, 'fiber_length': 140.4186699208927},
    22.0: {'source': 11.0, 'destination': 12.0, 'fiber_length': 124.3279883540976},
    23.0: {'source': 11.0, 'destination': 20.0, 'fiber_length': 83.74668764542909},
    24.0: {'source': 12.0, 'destination': 19.0, 'fiber_length': 78.96709458867502},
    25.0: {'source': 13.0, 'destination': 14.0, 'fiber_length': 116.6108708170878},
    26.0: {'source': 13.0, 'destination': 21.0, 'fiber_length': 122.0356871064491},
    27.0: {'source': 16.0, 'destination': 26.0, 'fiber_length': 219.890832991861},
    28.0: {'source': 17.0, 'destination': 28.0, 'fiber_length': 154.7754762812488},
    29.0: {'source': 17.0, 'destination': 18.0, 'fiber_length': 70.85134399759427},
    30.0: {'source': 18.0, 'destination': 19.0, 'fiber_length': 91.69810121883457},
    31.0: {'source': 20.0, 'destination': 37.0, 'fiber_length': 142.4138709656276},
    32.0: {'source': 21.0, 'destination': 28.0, 'fiber_length': 120.4615370867209},
    33.0: {'source': 22.0, 'destination': 24.0, 'fiber_length': 488.2417055105359},
    34.0: {'source': 23.0, 'destination': 24.0, 'fiber_length': 79.02061456911069},
    35.0: {'source': 23.0, 'destination': 25.0, 'fiber_length': 183.2872699483772},
    36.0: {'source': 24.0, 'destination': 25.0, 'fiber_length': 235.8713173227889},
    37.0: {'source': 24.0, 'destination': 26.0, 'fiber_length': 413.8389675135418},
    38.0: {'source': 26.0, 'destination': 27.0, 'fiber_length': 589.2508599924295},
    39.0: {'source': 27.0, 'destination': 32.0, 'fiber_length': 195.388303552465},
    40.0: {'source': 27.0, 'destination': 33.0, 'fiber_length': 108.5465409911054},
    41.0: {'source': 27.0, 'destination': 34.0, 'fiber_length': 306.8625058255422},
    42.0: {'source': 27.0, 'destination': 35.0, 'fiber_length': 168.1761860997829},
    43.0: {'source': 28.0, 'destination': 37.0, 'fiber_length': 144.1831991731319},
    44.0: {'source': 30.0, 'destination': 31.0, 'fiber_length': 171.6753296158936},
    45.0: {'source': 31.0, 'destination': 32.0, 'fiber_length': 255.9217977787615},
    46.0: {'source': 33.0, 'destination': 34.0, 'fiber_length': 203.4504493895497},
    47.0: {'source': 35.0, 'destination': 36.0, 'fiber_length': 164.130662797804},
    48.0: {'source': 36.0, 'destination': 37.0, 'fiber_length': 235.4621195478617},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
