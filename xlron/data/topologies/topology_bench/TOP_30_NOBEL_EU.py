def create_nobel_eu_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {
            "lat": 52.2,
            "long": 4.51,
            "location": "Sassenheim",
            "country": "Netherlands (Kingdom of the)",
        },
        2: {"lat": 37.58, "long": 23.42, "location": "Poros", "country": "Greece"},
        3: {"lat": 41.22, "long": 2.07, "location": "Viladecans", "country": "Spain"},
        4: {"lat": 44.47, "long": 20.25, "location": "Stepojevac", "country": "Serbia"},
        5: {"lat": 52.31, "long": 13.21, "location": "Ludwigsfelde", "country": "Germany"},
        6: {"lat": 44.51, "long": -0.35, "location": "Noaillan", "country": "France"},
        7: {"lat": 50.49, "long": 4.2, "location": "La Louviere", "country": "Belgium"},
        8: {"lat": 47.29, "long": 19.02, "location": "Szigethalom", "country": "Hungary"},
        9: {"lat": 55.41, "long": 12.32, "location": "Stroby Egede", "country": "Denmark"},
        10: {"lat": 53.21, "long": -6.15, "location": "Enniskerry", "country": "Ireland"},
        11: {"lat": 50.07, "long": 8.38, "location": "Hochheim am Main", "country": "Germany"},
        12: {
            "lat": 55.51,
            "long": -4.16,
            "location": "Muirkirk",
            "country": "United Kingdom of Great Britain and Northern Ireland (the)",
        },
        13: {"lat": 53.33, "long": 10.0, "location": "Marxen", "country": "Germany"},
        14: {
            "lat": 51.3,
            "long": -0.07,
            "location": "Whyteleafe",
            "country": "United Kingdom of Great Britain and Northern Ireland (the)",
        },
        15: {"lat": 45.43, "long": 4.49, "location": "Saint-Jean-Bonnefonds", "country": "France"},
        16: {"lat": 40.25, "long": -3.42, "location": "Morata de Tajuna", "country": "Spain"},
        17: {"lat": 45.28, "long": 9.11, "location": "Giovenzano", "country": "Italy"},
        18: {"lat": 48.07, "long": 11.33, "location": "Gauting", "country": "Germany"},
        19: {"lat": 59.54, "long": 10.45, "location": "Berger", "country": "Norway"},
        20: {"lat": 48.51, "long": 2.2, "location": "Etrechy", "country": "France"},
        21: {"lat": 50.04, "long": 14.25, "location": "Rudna", "country": "Czechia"},
        22: {"lat": 41.53, "long": 12.29, "location": "Torvaianica", "country": "Italy"},
        23: {"lat": 59.19, "long": 18.03, "location": "Huddinge", "country": "Sweden"},
        24: {"lat": 48.35, "long": 7.45, "location": "Epfig", "country": "France"},
        25: {"lat": 48.12, "long": 16.21, "location": "Kaltenleutgeben", "country": "Austria"},
        26: {"lat": 52.14, "long": 20.59, "location": "Bieniewice", "country": "Poland"},
        27: {"lat": 45.48, "long": 15.57, "location": "Karlovac", "country": "Croatia"},
        28: {"lat": 47.22, "long": 8.32, "location": "Hohenrain", "country": "Switzerland"},
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 7.0, "fiber_length": 287.0369913926625},
        2.0: {"source": 1.0, "destination": 12.0, "fiber_length": 1014.924288498928},
        3.0: {"source": 1.0, "destination": 13.0, "fiber_length": 585.0701325894771},
        4.0: {"source": 1.0, "destination": 14.0, "fiber_length": 496.082590705949},
        5.0: {"source": 2.0, "destination": 4.0, "fiber_length": 1216.193134088645},
        6.0: {"source": 2.0, "destination": 22.0, "fiber_length": 1500.0},
        7.0: {"source": 3.0, "destination": 15.0, "fiber_length": 761.0207893336487},
        8.0: {"source": 3.0, "destination": 16.0, "fiber_length": 712.328011104343},
        9.0: {"source": 4.0, "destination": 8.0, "fiber_length": 491.5420793362368},
        10.0: {"source": 4.0, "destination": 27.0, "fiber_length": 577.2240484085744},
        11.0: {"source": 5.0, "destination": 9.0, "fiber_length": 524.4037894750307},
        12.0: {"source": 5.0, "destination": 13.0, "fiber_length": 365.5096518359057},
        13.0: {"source": 5.0, "destination": 18.0, "fiber_length": 735.0758208533219},
        14.0: {"source": 5.0, "destination": 21.0, "fiber_length": 393.9165847124962},
        15.0: {"source": 5.0, "destination": 26.0, "fiber_length": 754.2256003851683},
        16.0: {"source": 6.0, "destination": 16.0, "fiber_length": 804.7897660190185},
        17.0: {"source": 6.0, "destination": 20.0, "fiber_length": 728.4507289037214},
        18.0: {"source": 7.0, "destination": 11.0, "fiber_length": 450.941856404134},
        19.0: {"source": 7.0, "destination": 20.0, "fiber_length": 394.9344268018788},
        20.0: {"source": 8.0, "destination": 21.0, "fiber_length": 697.2434946645521},
        21.0: {"source": 8.0, "destination": 26.0, "fiber_length": 826.4188164439588},
        22.0: {"source": 9.0, "destination": 19.0, "fiber_length": 708.8985474726883},
        23.0: {"source": 10.0, "destination": 12.0, "fiber_length": 429.57515190986},
        24.0: {"source": 10.0, "destination": 14.0, "fiber_length": 697.4339795081776},
        25.0: {"source": 11.0, "destination": 13.0, "fiber_length": 568.9082458927351},
        26.0: {"source": 11.0, "destination": 18.0, "fiber_length": 463.8143014051812},
        27.0: {"source": 11.0, "destination": 24.0, "fiber_length": 304.2479271607037},
        28.0: {"source": 14.0, "destination": 20.0, "fiber_length": 525.308569510817},
        29.0: {"source": 15.0, "destination": 20.0, "fiber_length": 575.98939482729},
        30.0: {"source": 15.0, "destination": 28.0, "fiber_length": 532.5782980276541},
        31.0: {"source": 17.0, "destination": 18.0, "fiber_length": 530.1308034294885},
        32.0: {"source": 17.0, "destination": 22.0, "fiber_length": 734.5002801181136},
        33.0: {"source": 17.0, "destination": 28.0, "fiber_length": 336.1565723463755},
        34.0: {"source": 18.0, "destination": 25.0, "fiber_length": 543.6059291132768},
        35.0: {"source": 19.0, "destination": 23.0, "fiber_length": 646.5222073761438},
        36.0: {"source": 20.0, "destination": 24.0, "fiber_length": 581.5286628993272},
        37.0: {"source": 21.0, "destination": 25.0, "fiber_length": 385.2039046334942},
        38.0: {"source": 22.0, "destination": 27.0, "fiber_length": 768.9337781286542},
        39.0: {"source": 23.0, "destination": 26.0, "fiber_length": 1200.066296064601},
        40.0: {"source": 24.0, "destination": 28.0, "fiber_length": 212.1978065100511},
        41.0: {"source": 25.0, "destination": 27.0, "fiber_length": 446.3493354434656},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
