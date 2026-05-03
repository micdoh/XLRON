def create_hibernia_us_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {
            "lat": 40.44,
            "long": -80.0,
            "location": "Pittsburgh",
            "country": "United States of America (the)",
        },
        2: {
            "lat": 39.95,
            "long": -75.16,
            "location": "Philadelphia",
            "country": "United States of America (the)",
        },
        3: {
            "lat": 40.74,
            "long": -74.17,
            "location": "Newark",
            "country": "United States of America (the)",
        },
        4: {
            "lat": 39.04,
            "long": -77.49,
            "location": "Ashburn",
            "country": "United States of America (the)",
        },
        5: {
            "lat": 35.77,
            "long": -78.64,
            "location": "Raleigh",
            "country": "United States of America (the)",
        },
        6: {
            "lat": 35.23,
            "long": -80.84,
            "location": "Charlotte",
            "country": "United States of America (the)",
        },
        7: {
            "lat": 38.93,
            "long": -77.18,
            "location": "McLean",
            "country": "United States of America (the)",
        },
        8: {
            "lat": 37.55,
            "long": -77.46,
            "location": "Richmond",
            "country": "United States of America (the)",
        },
        9: {
            "lat": 33.75,
            "long": -84.39,
            "location": "Atlanta",
            "country": "United States of America (the)",
        },
        10: {
            "lat": 40.71,
            "long": -74.01,
            "location": "New York City",
            "country": "United States of America (the)",
        },
        11: {
            "lat": 41.03,
            "long": -73.76,
            "location": "White Plains",
            "country": "United States of America (the)",
        },
        12: {
            "lat": 41.05,
            "long": -73.54,
            "location": "Stamford",
            "country": "United States of America (the)",
        },
        13: {
            "lat": 41.85,
            "long": -87.65,
            "location": "Chicago",
            "country": "United States of America (the)",
        },
        14: {
            "lat": 41.5,
            "long": -81.7,
            "location": "Cleveland",
            "country": "United States of America (the)",
        },
        15: {
            "lat": 42.89,
            "long": -78.88,
            "location": "Buffalo",
            "country": "United States of America (the)",
        },
        16: {"lat": 43.7, "long": -79.42, "location": "Toronto", "country": "Canada"},
        17: {"lat": 45.51, "long": -73.59, "location": "Montreal", "country": "Canada"},
        18: {
            "lat": 42.65,
            "long": -73.76,
            "location": "Albany",
            "country": "United States of America (the)",
        },
        19: {
            "lat": 42.36,
            "long": -71.06,
            "location": "Boston",
            "country": "United States of America (the)",
        },
        20: {"lat": 44.65, "long": -63.57, "location": "Halifax", "country": "Canada"},
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 4.0, "fiber_length": 397.6649920225643},
        2.0: {"source": 1.0, "destination": 13.0, "fiber_length": 988.8431894766425},
        3.0: {"source": 1.0, "destination": 14.0, "fiber_length": 277.6478760498783},
        4.0: {"source": 2.0, "destination": 10.0, "fiber_length": 193.5165266446945},
        5.0: {"source": 2.0, "destination": 3.0, "fiber_length": 182.2082908548479},
        6.0: {"source": 2.0, "destination": 4.0, "fiber_length": 336.1009376453503},
        7.0: {"source": 3.0, "destination": 4.0, "fiber_length": 510.74493402578},
        8.0: {"source": 3.0, "destination": 11.0, "fiber_length": 70.79952412600834},
        9.0: {"source": 3.0, "destination": 10.0, "fiber_length": 20.83436718327859},
        10.0: {"source": 4.0, "destination": 7.0, "fiber_length": 44.18098870812758},
        11.0: {"source": 5.0, "destination": 6.0, "fiber_length": 312.0079821457963},
        12.0: {"source": 5.0, "destination": 8.0, "fiber_length": 336.2521540699969},
        13.0: {"source": 6.0, "destination": 9.0, "fiber_length": 546.8613702582421},
        14.0: {"source": 7.0, "destination": 8.0, "fiber_length": 233.0775281483698},
        15.0: {"source": 10.0, "destination": 18.0, "fiber_length": 325.0720000910152},
        16.0: {"source": 10.0, "destination": 12.0, "fiber_length": 82.03035170425309},
        17.0: {"source": 11.0, "destination": 12.0, "fiber_length": 27.87704634020835},
        18.0: {"source": 12.0, "destination": 19.0, "fiber_length": 378.2724731274459},
        19.0: {"source": 13.0, "destination": 14.0, "fiber_length": 743.4071200946746},
        20.0: {"source": 14.0, "destination": 15.0, "fiber_length": 418.5080080999496},
        21.0: {"source": 15.0, "destination": 16.0, "fiber_length": 150.1652468643198},
        22.0: {"source": 15.0, "destination": 18.0, "fiber_length": 628.0712841529555},
        23.0: {"source": 16.0, "destination": 17.0, "fiber_length": 755.0357670265992},
        24.0: {"source": 17.0, "destination": 18.0, "fiber_length": 477.4605770094612},
        25.0: {"source": 17.0, "destination": 20.0, "fiber_length": 1187.999858185761},
        26.0: {"source": 18.0, "destination": 19.0, "fiber_length": 335.4884708932951},
        27.0: {"source": 19.0, "destination": 20.0, "fiber_length": 982.8418925334275},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
