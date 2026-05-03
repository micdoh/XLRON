def create_kentmanfeb2008_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {
            "lat": 51.28,
            "long": 1.09,
            "location": "Canterbury",
            "country": "United Kingdom of Great Britain and Northern Ireland (the)",
        },
        2: {
            "lat": 51.27,
            "long": 1.09,
            "location": "Canterbury",
            "country": "United Kingdom of Great Britain and Northern Ireland (the)",
        },
        3: {
            "lat": 51.39,
            "long": 0.55,
            "location": "Gillingham",
            "country": "United Kingdom of Great Britain and Northern Ireland (the)",
        },
        4: {
            "lat": 51.39,
            "long": 0.55,
            "location": "Gillingham",
            "country": "United Kingdom of Great Britain and Northern Ireland (the)",
        },
        5: {
            "lat": 51.14,
            "long": 0.86,
            "location": "Ashford",
            "country": "United Kingdom of Great Britain and Northern Ireland (the)",
        },
        6: {
            "lat": 51.27,
            "long": 0.5,
            "location": "Maidstone",
            "country": "United Kingdom of Great Britain and Northern Ireland (the)",
        },
        7: {
            "lat": 51.35,
            "long": 1.43,
            "location": "Broadstairs",
            "country": "United Kingdom of Great Britain and Northern Ireland (the)",
        },
        8: {
            "lat": 51.36,
            "long": 1.43,
            "location": "Broadstairs",
            "country": "United Kingdom of Great Britain and Northern Ireland (the)",
        },
        9: {
            "lat": 51.28,
            "long": 0.52,
            "location": "Maidstone",
            "country": "United Kingdom of Great Britain and Northern Ireland (the)",
        },
        10: {
            "lat": 51.22,
            "long": 0.33,
            "location": "Hadlow",
            "country": "United Kingdom of Great Britain and Northern Ireland (the)",
        },
        11: {
            "lat": 51.08,
            "long": 1.18,
            "location": "Folkestone",
            "country": "United Kingdom of Great Britain and Northern Ireland (the)",
        },
        12: {
            "lat": 51.3,
            "long": 1.06,
            "location": "Blean",
            "country": "United Kingdom of Great Britain and Northern Ireland (the)",
        },
        13: {
            "lat": 51.19,
            "long": 0.26,
            "location": "Tonbridge",
            "country": "United Kingdom of Great Britain and Northern Ireland (the)",
        },
        14: {
            "lat": 51.44,
            "long": 0.03,
            "location": "Blackheath",
            "country": "United Kingdom of Great Britain and Northern Ireland (the)",
        },
        15: {
            "lat": 51.28,
            "long": 1.08,
            "location": "Canterbury",
            "country": "United Kingdom of Great Britain and Northern Ireland (the)",
        },
        16: {
            "lat": 51.45,
            "long": 0.22,
            "location": "Dartford",
            "country": "United Kingdom of Great Britain and Northern Ireland (the)",
        },
        17: {
            "lat": 51.38,
            "long": 0.51,
            "location": "Rochester",
            "country": "United Kingdom of Great Britain and Northern Ireland (the)",
        },
        18: {
            "lat": 51.45,
            "long": 0.08,
            "location": "Welling",
            "country": "United Kingdom of Great Britain and Northern Ireland (the)",
        },
        19: {
            "lat": 51.45,
            "long": 0.08,
            "location": "Welling",
            "country": "United Kingdom of Great Britain and Northern Ireland (the)",
        },
        20: {
            "lat": 51.28,
            "long": 1.09,
            "location": "Canterbury",
            "country": "United Kingdom of Great Britain and Northern Ireland (the)",
        },
        21: {
            "lat": 51.18,
            "long": 0.94,
            "location": "Wye",
            "country": "United Kingdom of Great Britain and Northern Ireland (the)",
        },
        22: {
            "lat": 51.26,
            "long": 0.55,
            "location": "Maidstone",
            "country": "United Kingdom of Great Britain and Northern Ireland (the)",
        },
        23: {
            "lat": 51.15,
            "long": 0.24,
            "location": "Royal Tunbridge Wells",
            "country": "United Kingdom of Great Britain and Northern Ireland (the)",
        },
        24: {
            "lat": 51.36,
            "long": 1.41,
            "location": "Ramsgate",
            "country": "United Kingdom of Great Britain and Northern Ireland (the)",
        },
        25: {
            "lat": 51.29,
            "long": 0.22,
            "location": "Kemsing",
            "country": "United Kingdom of Great Britain and Northern Ireland (the)",
        },
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 12.0, "fiber_length": 4.573850058655266},
        2.0: {"source": 2.0, "destination": 12.0, "fiber_length": 5.901871471914736},
        3.0: {"source": 3.0, "destination": 19.0, "fiber_length": 49.89976819615696},
        4.0: {"source": 4.0, "destination": 19.0, "fiber_length": 49.89976819615696},
        5.0: {"source": 5.0, "destination": 21.0, "fiber_length": 10.70231608645681},
        6.0: {"source": 6.0, "destination": 22.0, "fiber_length": 5.478340377084016},
        7.0: {"source": 7.0, "destination": 24.0, "fiber_length": 2.668661059794504},
        8.0: {"source": 8.0, "destination": 24.0, "fiber_length": 2.082986918591846},
        9.0: {"source": 9.0, "destination": 19.0, "fiber_length": 53.88424397417296},
        10.0: {"source": 10.0, "destination": 23.0, "fiber_length": 14.99498884779001},
        11.0: {"source": 11.0, "destination": 21.0, "fiber_length": 30.15407073351049},
        12.0: {"source": 12.0, "destination": 15.0, "fiber_length": 3.93446057915911},
        13.0: {"source": 12.0, "destination": 19.0, "fiber_length": 105.0546214596761},
        14.0: {"source": 12.0, "destination": 20.0, "fiber_length": 4.573850058655266},
        15.0: {"source": 12.0, "destination": 21.0, "fiber_length": 23.61398311511024},
        16.0: {"source": 12.0, "destination": 24.0, "fiber_length": 37.82402118280783},
        17.0: {"source": 13.0, "destination": 23.0, "fiber_length": 6.991879443216483},
        18.0: {"source": 14.0, "destination": 18.0, "fiber_length": 5.458852644411509},
        19.0: {"source": 16.0, "destination": 18.0, "fiber_length": 14.5522384383509},
        20.0: {"source": 17.0, "destination": 19.0, "fiber_length": 46.22899646916576},
        21.0: {"source": 18.0, "destination": 19.0, "fiber_length": 0.0},
        22.0: {"source": 19.0, "destination": 22.0, "fiber_length": 58.31736466329251},
        23.0: {"source": 19.0, "destination": 25.0, "fiber_length": 30.40877451609891},
        24.0: {"source": 21.0, "destination": 22.0, "fiber_length": 42.87159291906983},
        25.0: {"source": 22.0, "destination": 23.0, "fiber_length": 37.230084913817},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
