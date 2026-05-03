def create_pionier21_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {"lat": 53.4301818, "long": 14.5509623, "location": "Szcezin", "country": "Poland"},
        2: {
            "lat": 54.20717985,
            "long": 16.2175410869958,
            "location": "Koszalin",
            "country": "Poland",
        },
        3: {
            "lat": 54.52333035,
            "long": 18.6040278920418,
            "location": "Gdynia",
            "country": "Poland",
        },
        4: {"lat": 53.0145361, "long": 18.5965831446515, "location": "Thorn", "country": "Poland"},
        5: {
            "lat": 53.12974625,
            "long": 18.0293696585348,
            "location": "Bydgoszcz",
            "country": "Poland",
        },
        6: {"lat": 52.4006632, "long": 16.9197325917808, "location": "Poznan", "country": "Poland"},
        7: {"lat": 51.9383777, "long": 15.5050408, "location": "Zielona Gora", "country": "Poland"},
        8: {
            "lat": 51.1263106,
            "long": 16.9781963305126,
            "location": "Wroclaw",
            "country": "Poland",
        },
        9: {"lat": 50.6787929, "long": 17.9298844360335, "location": "Opole", "country": "Poland"},
        10: {
            "lat": 50.30113145,
            "long": 18.6623472279719,
            "location": "Gliwice",
            "country": "Poland",
        },
        11: {
            "lat": 50.8089997,
            "long": 19.1244089281233,
            "location": "Chestokhov",
            "country": "Poland",
        },
        12: {"lat": 51.7728245, "long": 19.4784859313079, "location": "Lodz", "country": "Poland"},
        13: {"lat": 52.2319581, "long": 21.0067249, "location": "Warsaw", "country": "Poland"},
        14: {"lat": 53.132398, "long": 23.1591679, "location": "Belostok", "country": "Poland"},
        15: {
            "lat": 53.77664395,
            "long": 20.4777530921921,
            "location": "Olsztyn",
            "country": "Poland",
        },
        16: {
            "lat": 50.0469432,
            "long": 19.9971534358366,
            "location": "Cracow",
            "country": "Poland",
        },
        17: {
            "lat": 50.85402845,
            "long": 20.6099156873451,
            "location": "Kielce",
            "country": "Poland",
        },
        18: {"lat": 51.4022557, "long": 21.1541546, "location": "Radom", "country": "Poland"},
        19: {
            "lat": 51.56723175,
            "long": 21.856244041776,
            "location": "Deblin",
            "country": "Poland",
        },
        20: {
            "lat": 51.2181944999999,
            "long": 22.5546775621451,
            "location": "Lublin",
            "country": "Poland",
        },
        21: {
            "lat": 50.013064,
            "long": 22.0161434572943,
            "location": "Rzeszow",
            "country": "Poland",
        },
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 2.0, "fiber_length": 209.1},
        2.0: {"source": 1.0, "destination": 6.0, "fiber_length": 293.655},
        3.0: {"source": 2.0, "destination": 3.0, "fiber_length": 237.81},
        4.0: {"source": 3.0, "destination": 15.0, "fiber_length": 221.37},
        5.0: {"source": 3.0, "destination": 4.0, "fiber_length": 251.655},
        6.0: {"source": 4.0, "destination": 5.0, "fiber_length": 60.0},
        7.0: {"source": 5.0, "destination": 6.0, "fiber_length": 165.315},
        8.0: {"source": 6.0, "destination": 13.0, "fiber_length": 417.6},
        9.0: {"source": 6.0, "destination": 7.0, "fiber_length": 163.98},
        10.0: {"source": 7.0, "destination": 8.0, "fiber_length": 204.225},
        11.0: {"source": 8.0, "destination": 9.0, "fiber_length": 124.875},
        12.0: {"source": 9.0, "destination": 10.0, "fiber_length": 100.05},
        13.0: {"source": 10.0, "destination": 11.0, "fiber_length": 97.845},
        14.0: {"source": 10.0, "destination": 16.0, "fiber_length": 148.755},
        15.0: {"source": 11.0, "destination": 12.0, "fiber_length": 164.94},
        16.0: {"source": 12.0, "destination": 13.0, "fiber_length": 174.6},
        17.0: {"source": 13.0, "destination": 14.0, "fiber_length": 264.42},
        18.0: {"source": 13.0, "destination": 18.0, "fiber_length": 139.215},
        19.0: {"source": 14.0, "destination": 15.0, "fiber_length": 287.16},
        20.0: {"source": 15.0, "destination": 3.0, "fiber_length": 221.37},
        21.0: {"source": 16.0, "destination": 17.0, "fiber_length": 149.52},
        22.0: {"source": 16.0, "destination": 21.0, "fiber_length": 216.39},
        23.0: {"source": 17.0, "destination": 18.0, "fiber_length": 107.73},
        24.0: {"source": 18.0, "destination": 19.0, "fiber_length": 77.94},
        25.0: {"source": 19.0, "destination": 20.0, "fiber_length": 93.135},
        26.0: {"source": 20.0, "destination": 21.0, "fiber_length": 208.935},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
