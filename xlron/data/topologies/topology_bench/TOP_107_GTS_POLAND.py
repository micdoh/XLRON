def create_gts_poland_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {"lat": 52.23, "long": 21.01, "location": "Warsaw", "country": "Poland"},
        2: {"lat": 53.13, "long": 23.15, "location": "Bialystok", "country": "Poland"},
        3: {"lat": 52.65, "long": 19.07, "location": "Wloclawek", "country": "Poland"},
        4: {"lat": 52.55, "long": 19.71, "location": "Plock", "country": "Poland"},
        5: {"lat": 50.67, "long": 17.95, "location": "Opole", "country": "Poland"},
        6: {"lat": 51.1, "long": 17.03, "location": "Wroclaw", "country": "Poland"},
        7: {"lat": 51.25, "long": 22.57, "location": "Lublin", "country": "Poland"},
        8: {"lat": 50.04, "long": 22.0, "location": "Rzeszow", "country": "Poland"},
        9: {"lat": 51.76, "long": 18.09, "location": "Kalisz", "country": "Poland"},
        10: {"lat": 51.75, "long": 19.47, "location": "Lodz", "country": "Poland"},
        11: {"lat": 50.08, "long": 19.92, "location": "Krakow", "country": "Poland"},
        12: {"lat": 51.94, "long": 15.51, "location": "Zielona Gora", "country": "Poland"},
        13: {"lat": 54.35, "long": 18.65, "location": "Gdansk", "country": "Poland"},
        14: {"lat": 53.78, "long": 20.48, "location": "Olsztyn", "country": "Poland"},
        15: {"lat": 50.01, "long": 20.99, "location": "Tarnow", "country": "Poland"},
        16: {"lat": 50.27, "long": 19.02, "location": "Katowice", "country": "Poland"},
        17: {"lat": 50.87, "long": 20.63, "location": "Kielce", "country": "Poland"},
        18: {"lat": 51.4, "long": 21.15, "location": "Radom", "country": "Poland"},
        19: {"lat": 53.12, "long": 18.01, "location": "Bydgoszcz", "country": "Poland"},
        20: {"lat": 53.01, "long": 18.6, "location": "Torun", "country": "Poland"},
        21: {"lat": 54.46, "long": 17.03, "location": "Slupsk", "country": "Poland"},
        22: {"lat": 54.19, "long": 16.17, "location": "Koszalin", "country": "Poland"},
        23: {"lat": 54.18, "long": 15.58, "location": "Kolobrzeg", "country": "Poland"},
        24: {"lat": 53.43, "long": 14.55, "location": "Szczecin", "country": "Poland"},
        25: {"lat": 52.42, "long": 16.97, "location": "Poznan", "country": "Poland"},
        26: {"lat": 53.15, "long": 16.74, "location": "Pila", "country": "Poland"},
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 2.0, "fiber_length": 263.3489639066166},
        2.0: {"source": 1.0, "destination": 4.0, "fiber_length": 142.6837166878209},
        3.0: {"source": 1.0, "destination": 7.0, "fiber_length": 229.5070966428595},
        4.0: {"source": 1.0, "destination": 10.0, "fiber_length": 177.2765341606382},
        5.0: {"source": 1.0, "destination": 18.0, "fiber_length": 139.1882096147356},
        6.0: {"source": 2.0, "destination": 14.0, "fiber_length": 286.4603308844971},
        7.0: {"source": 3.0, "destination": 20.0, "fiber_length": 76.4765358424114},
        8.0: {"source": 3.0, "destination": 4.0, "fiber_length": 66.9464074833659},
        9.0: {"source": 5.0, "destination": 16.0, "fiber_length": 131.7334387463687},
        10.0: {"source": 5.0, "destination": 6.0, "fiber_length": 120.4791984364237},
        11.0: {"source": 6.0, "destination": 9.0, "fiber_length": 155.7813418938423},
        12.0: {"source": 7.0, "destination": 8.0, "fiber_length": 210.6290855812812},
        13.0: {"source": 8.0, "destination": 11.0, "fiber_length": 222.8153948007431},
        14.0: {"source": 9.0, "destination": 10.0, "fiber_length": 142.4908821143913},
        15.0: {"source": 9.0, "destination": 25.0, "fiber_length": 159.0324055171153},
        16.0: {"source": 11.0, "destination": 17.0, "fiber_length": 151.7951398283662},
        17.0: {"source": 11.0, "destination": 15.0, "fiber_length": 115.2016707785395},
        18.0: {"source": 12.0, "destination": 25.0, "fiber_length": 169.4246184805271},
        19.0: {"source": 13.0, "destination": 14.0, "fiber_length": 202.7866550232714},
        20.0: {"source": 13.0, "destination": 19.0, "fiber_length": 214.6495959556784},
        21.0: {"source": 17.0, "destination": 18.0, "fiber_length": 103.8090366128347},
        22.0: {"source": 19.0, "destination": 20.0, "fiber_length": 61.91455731777963},
        23.0: {"source": 19.0, "destination": 26.0, "fiber_length": 127.1781145390214},
        24.0: {"source": 21.0, "destination": 22.0, "fiber_length": 95.00385971776764},
        25.0: {"source": 22.0, "destination": 23.0, "fiber_length": 57.60910427743721},
        26.0: {"source": 23.0, "destination": 24.0, "fiber_length": 161.058611099911},
        27.0: {"source": 24.0, "destination": 26.0, "fiber_length": 223.2777237940912},
        28.0: {"source": 25.0, "destination": 26.0, "fiber_length": 123.9491665121685},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
