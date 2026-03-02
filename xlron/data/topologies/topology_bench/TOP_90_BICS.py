def create_bics_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 48.15, 'long': 17.11, 'location': 'Bratislava', 'country': 'Slovakia'},
    2: {'lat': 48.21, 'long': 16.37, 'location': 'Vienna', 'country': 'Austria'},
    3: {'lat': 50.09, 'long': 14.42, 'location': 'Prague', 'country': 'Czechia'},
    4: {'lat': 51.92, 'long': 4.48, 'location': 'Rotterdam', 'country': 'Netherlands (Kingdom of the)'},
    5: {'lat': 41.89, 'long': 12.48, 'location': 'Vatican City', 'country': 'Holy See (the)'},
    6: {'lat': 47.5, 'long': 19.04, 'location': 'Budapest', 'country': 'Hungary'},
    7: {'lat': 46.05, 'long': 14.51, 'location': 'Ljubljana', 'country': 'Slovenia'},
    8: {'lat': 45.81, 'long': 15.98, 'location': 'Zagreb - Centar', 'country': 'Croatia'},
    9: {'lat': 44.43, 'long': 26.11, 'location': 'Bucharest', 'country': 'Romania'},
    10: {'lat': 41.01, 'long': 28.95, 'location': 'Istanbul', 'country': 'Türkiye'},
    11: {'lat': 47.14, 'long': 9.52, 'location': 'Vaduz', 'country': 'Liechtenstein'},
    12: {'lat': 48.58, 'long': 7.75, 'location': 'Strasbourg', 'country': 'France'},
    13: {'lat': 49.61, 'long': 6.13, 'location': 'Luxembourg', 'country': 'Luxembourg'},
    14: {'lat': 47.37, 'long': 8.55, 'location': 'Zurich', 'country': 'Switzerland'},
    15: {'lat': 50.85, 'long': 4.35, 'location': 'Brussels', 'country': 'Belgium'},
    16: {'lat': 46.2, 'long': 6.15, 'location': 'Geneve', 'country': 'Switzerland'},
    17: {'lat': 45.46, 'long': 9.19, 'location': 'Milano', 'country': 'Italy'},
    18: {'lat': 37.98, 'long': 23.72, 'location': 'Athens', 'country': 'Greece'},
    19: {'lat': 42.7, 'long': 23.32, 'location': 'Sofia', 'country': 'Bulgaria'},
    20: {'lat': 50.12, 'long': 8.68, 'location': 'Frankfurt am Main', 'country': 'Germany'},
    21: {'lat': 52.37, 'long': 4.89, 'location': 'Amsterdam', 'country': 'Netherlands (Kingdom of the)'},
    22: {'lat': 51.51, 'long': -0.13, 'location': 'London', 'country': 'United Kingdom of Great Britain and Northern Ireland (the)'},
    23: {'lat': 48.85, 'long': 2.35, 'location': 'Paris', 'country': 'France'},
    24: {'lat': 59.33, 'long': 18.06, 'location': 'Stockholm', 'country': 'Sweden'},
    25: {'lat': 52.23, 'long': 21.01, 'location': 'Warsaw', 'country': 'Poland'},
    26: {'lat': 50.45, 'long': 30.52, 'location': 'Kiev', 'country': 'Ukraine'},
    27: {'lat': 53.34, 'long': -6.27, 'location': 'Rathmines', 'country': 'Ireland'},
    28: {'lat': 38.72, 'long': -9.13, 'location': 'Lisbon', 'country': 'Portugal'},
    29: {'lat': 40.42, 'long': -3.7, 'location': 'City Center', 'country': 'Spain'},
    30: {'lat': 41.39, 'long': 2.16, 'location': 'Barcelona', 'country': 'Spain'},
    31: {'lat': 43.3, 'long': 5.38, 'location': 'Marseille', 'country': 'France'},
    32: {'lat': 45.75, 'long': 4.85, 'location': 'Lyon', 'country': 'France'},
    33: {'lat': 47.57, 'long': 7.6, 'location': 'Birsfelden', 'country': 'Switzerland'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 82.90568141920215},
    2.0: {'source': 1.0, 'destination': 3.0, 'fiber_length': 436.8975089563237},
    3.0: {'source': 1.0, 'destination': 6.0, 'fiber_length': 241.7867396523617},
    4.0: {'source': 2.0, 'destination': 17.0, 'fiber_length': 938.3690097105002},
    5.0: {'source': 2.0, 'destination': 20.0, 'fiber_length': 896.6393352298046},
    6.0: {'source': 3.0, 'destination': 25.0, 'fiber_length': 775.8606332236109},
    7.0: {'source': 3.0, 'destination': 20.0, 'fiber_length': 613.9214384057149},
    8.0: {'source': 4.0, 'destination': 21.0, 'fiber_length': 85.99144004903886},
    9.0: {'source': 4.0, 'destination': 15.0, 'fiber_length': 178.9800585255622},
    10.0: {'source': 5.0, 'destination': 17.0, 'fiber_length': 715.4516592483906},
    11.0: {'source': 5.0, 'destination': 14.0, 'fiber_length': 1025.843209642385},
    12.0: {'source': 6.0, 'destination': 9.0, 'fiber_length': 965.8917706148927},
    13.0: {'source': 6.0, 'destination': 19.0, 'fiber_length': 945.6352538534798},
    14.0: {'source': 6.0, 'destination': 8.0, 'fiber_length': 449.58607573109},
    15.0: {'source': 7.0, 'destination': 8.0, 'fiber_length': 175.1673780561866},
    16.0: {'source': 9.0, 'destination': 26.0, 'fiber_length': 1120.089078640492},
    17.0: {'source': 9.0, 'destination': 10.0, 'fiber_length': 668.1031392122177},
    18.0: {'source': 10.0, 'destination': 18.0, 'fiber_length': 841.4865647688955},
    19.0: {'source': 11.0, 'destination': 14.0, 'fiber_length': 116.3188714196001},
    20.0: {'source': 12.0, 'destination': 33.0, 'fiber_length': 169.2875823684238},
    21.0: {'source': 12.0, 'destination': 20.0, 'fiber_length': 276.0159872835782},
    22.0: {'source': 12.0, 'destination': 13.0, 'fiber_length': 246.6032536389703},
    23.0: {'source': 13.0, 'destination': 15.0, 'fiber_length': 280.7794512078589},
    24.0: {'source': 14.0, 'destination': 17.0, 'fiber_length': 326.9602959297368},
    25.0: {'source': 14.0, 'destination': 20.0, 'fiber_length': 458.9016589907375},
    26.0: {'source': 14.0, 'destination': 16.0, 'fiber_length': 336.4474057053269},
    27.0: {'source': 15.0, 'destination': 20.0, 'fiber_length': 475.3066139514598},
    28.0: {'source': 15.0, 'destination': 21.0, 'fiber_length': 259.619106832056},
    29.0: {'source': 15.0, 'destination': 22.0, 'fiber_length': 481.0970345108332},
    30.0: {'source': 15.0, 'destination': 23.0, 'fiber_length': 396.8832465221968},
    31.0: {'source': 16.0, 'destination': 33.0, 'fiber_length': 282.0120443677531},
    32.0: {'source': 16.0, 'destination': 23.0, 'fiber_length': 615.1060613002187},
    33.0: {'source': 16.0, 'destination': 32.0, 'fiber_length': 168.3455938030974},
    34.0: {'source': 17.0, 'destination': 31.0, 'fiber_length': 579.6120453898972},
    35.0: {'source': 18.0, 'destination': 19.0, 'fiber_length': 788.8978423686046},
    36.0: {'source': 20.0, 'destination': 21.0, 'fiber_length': 545.2339470871901},
    37.0: {'source': 20.0, 'destination': 24.0, 'fiber_length': 1500.0},
    38.0: {'source': 20.0, 'destination': 25.0, 'fiber_length': 1334.654553161583},
    39.0: {'source': 21.0, 'destination': 22.0, 'fiber_length': 535.6163051270169},
    40.0: {'source': 22.0, 'destination': 27.0, 'fiber_length': 694.7866744906818},
    41.0: {'source': 22.0, 'destination': 28.0, 'fiber_length': 1981.348079362193},
    42.0: {'source': 22.0, 'destination': 23.0, 'fiber_length': 516.656454570689},
    43.0: {'source': 23.0, 'destination': 29.0, 'fiber_length': 1500.0},
    44.0: {'source': 23.0, 'destination': 32.0, 'fiber_length': 589.2508599924295},
    45.0: {'source': 25.0, 'destination': 26.0, 'fiber_length': 1033.531903854255},
    46.0: {'source': 29.0, 'destination': 30.0, 'fiber_length': 756.0643238408899},
    47.0: {'source': 30.0, 'destination': 31.0, 'fiber_length': 508.909806592813},
    48.0: {'source': 31.0, 'destination': 32.0, 'fiber_length': 413.4699996869407},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
