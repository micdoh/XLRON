def create_optunet_sweden_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {"lat": 57.70716, "long": 11.96679, "location": "Goeteborg", "country": "Sweden"},
        2: {"lat": 58.39118, "long": 13.84506, "location": "Skoevde", "country": "Sweden"},
        3: {"lat": 59.27412, "long": 15.2066, "location": "Orebro", "country": "Sweden"},
        4: {"lat": 59.3793, "long": 13.50357, "location": "Karlstad", "country": "Sweden"},
        5: {"lat": 56.67446, "long": 12.85676, "location": "Halmstad", "country": "Sweden"},
        6: {"lat": 58.41667, "long": 15.61667, "location": "Linkoping", "country": "Sweden"},
        7: {"lat": 57.72101, "long": 12.9401, "location": "Boras", "country": "Sweden"},
        8: {"lat": 58.28365, "long": 12.28864, "location": "Trollhattan", "country": "Sweden"},
        9: {"lat": 57.78145, "long": 14.15618, "location": "Jonkoping", "country": "Sweden"},
        10: {"lat": 56.86667, "long": 14.81667, "location": "Vaexjoe", "country": "Sweden"},
        11: {"lat": 55.70584, "long": 13.19321, "location": "Lund", "country": "Sweden"},
        12: {"lat": 56.03129, "long": 14.15242, "location": "Kristianstad", "country": "Sweden"},
        13: {"lat": 56.20999, "long": 15.27602, "location": "Ronneby", "country": "Sweden"},
        14: {"lat": 56.66157, "long": 16.36163, "location": "Kalmar", "country": "Sweden"},
        15: {"lat": 57.64089, "long": 18.29602, "location": "Visby", "country": "Sweden"},
        16: {"lat": 55.60587, "long": 13.00073, "location": "Malmoe", "country": "Sweden"},
        17: {"lat": 59.33258, "long": 18.0649, "location": "Stockholm", "country": "Sweden"},
        18: {"lat": 65.85, "long": 23.16667, "location": "Kalix", "country": "Sweden"},
        19: {"lat": 67.85572, "long": 20.22513, "location": "Kiruna", "country": "Sweden"},
        20: {"lat": 65.58415, "long": 22.15465, "location": "Lulea", "country": "Sweden"},
        21: {"lat": 63.82842, "long": 20.25972, "location": "Umea", "country": "Sweden"},
        22: {"lat": 62.39129, "long": 17.3063, "location": "Sundsvall", "country": "Sweden"},
        23: {"lat": 60.67452, "long": 17.14174, "location": "Gavle", "country": "Sweden"},
        24: {"lat": 60.4858, "long": 15.43714, "location": "Borlaenge", "country": "Sweden"},
        25: {"lat": 59.8585, "long": 17.64543, "location": "Uppsala", "country": "Sweden"},
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 5.0, "fiber_length": 190.0960808400053},
        2.0: {"source": 1.0, "destination": 7.0, "fiber_length": 86.74341799687855},
        3.0: {"source": 1.0, "destination": 8.0, "fiber_length": 100.2746698658031},
        4.0: {"source": 2.0, "destination": 3.0, "fiber_length": 188.4095708659947},
        5.0: {"source": 2.0, "destination": 7.0, "fiber_length": 137.3740662255246},
        6.0: {"source": 2.0, "destination": 8.0, "fiber_length": 137.4396981416263},
        7.0: {"source": 3.0, "destination": 24.0, "fiber_length": 203.0177338605977},
        8.0: {"source": 3.0, "destination": 4.0, "fiber_length": 145.9610522942691},
        9.0: {"source": 3.0, "destination": 18.0, "fiber_length": 1253.714379773383},
        10.0: {"source": 4.0, "destination": 8.0, "fiber_length": 210.694581270405},
        11.0: {"source": 5.0, "destination": 11.0, "fiber_length": 164.5478734830161},
        12.0: {"source": 6.0, "destination": 17.0, "fiber_length": 260.5348643627447},
        13.0: {"source": 6.0, "destination": 9.0, "fiber_length": 166.7174780120039},
        14.0: {"source": 9.0, "destination": 10.0, "fiber_length": 163.7588213038351},
        15.0: {"source": 10.0, "destination": 11.0, "fiber_length": 245.0906239697914},
        16.0: {"source": 10.0, "destination": 14.0, "fiber_length": 145.3156585583322},
        17.0: {"source": 11.0, "destination": 16.0, "fiber_length": 24.6185530101705},
        18.0: {"source": 12.0, "destination": 13.0, "fiber_length": 108.6370977105624},
        19.0: {"source": 12.0, "destination": 16.0, "fiber_length": 129.1553528994795},
        20.0: {"source": 13.0, "destination": 14.0, "fiber_length": 125.2771429335356},
        21.0: {"source": 15.0, "destination": 17.0, "fiber_length": 282.8790813503142},
        22.0: {"source": 17.0, "destination": 18.0, "fiber_length": 1154.573279465749},
        23.0: {"source": 17.0, "destination": 25.0, "fiber_length": 94.59605471963164},
        24.0: {"source": 19.0, "destination": 20.0, "fiber_length": 399.6140407572479},
        25.0: {"source": 20.0, "destination": 21.0, "fiber_length": 322.4443218961811},
        26.0: {"source": 21.0, "destination": 22.0, "fiber_length": 327.1913618719075},
        27.0: {"source": 22.0, "destination": 23.0, "fiber_length": 286.6426397341182},
        28.0: {"source": 23.0, "destination": 24.0, "fiber_length": 143.1556159924878},
        29.0: {"source": 23.0, "destination": 25.0, "fiber_length": 142.339911379905},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
