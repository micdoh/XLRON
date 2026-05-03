def create_sunet_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {"lat": 57.71, "long": 11.97, "location": "Goeteborg", "country": "Sweden"},
        2: {"lat": 58.39, "long": 13.85, "location": "Skoevde", "country": "Sweden"},
        3: {"lat": 59.27, "long": 15.21, "location": "Orebro", "country": "Sweden"},
        4: {"lat": 59.38, "long": 13.5, "location": "Karlstad", "country": "Sweden"},
        5: {"lat": 56.67, "long": 12.86, "location": "Halmstad", "country": "Sweden"},
        6: {"lat": 58.42, "long": 15.62, "location": "Linkoping", "country": "Sweden"},
        7: {"lat": 57.72, "long": 12.94, "location": "Boras", "country": "Sweden"},
        8: {"lat": 58.28, "long": 12.29, "location": "Trollhattan", "country": "Sweden"},
        9: {"lat": 57.78, "long": 14.16, "location": "Jonkoping", "country": "Sweden"},
        10: {"lat": 56.87, "long": 14.82, "location": "Vaexjoe", "country": "Sweden"},
        11: {"lat": 55.71, "long": 13.19, "location": "Lund", "country": "Sweden"},
        12: {"lat": 56.03, "long": 14.15, "location": "Kristianstad", "country": "Sweden"},
        13: {"lat": 56.21, "long": 15.28, "location": "Ronneby", "country": "Sweden"},
        14: {"lat": 56.66, "long": 16.36, "location": "Kalmar", "country": "Sweden"},
        15: {"lat": 57.64, "long": 18.3, "location": "Visby", "country": "Sweden"},
        16: {"lat": 55.61, "long": 13.0, "location": "Malmoe", "country": "Sweden"},
        17: {"lat": 59.33, "long": 18.06, "location": "Stockholm", "country": "Sweden"},
        18: {"lat": 65.85, "long": 23.17, "location": "Kalix", "country": "Sweden"},
        19: {"lat": 67.86, "long": 20.23, "location": "Kiruna", "country": "Sweden"},
        20: {"lat": 65.58, "long": 22.15, "location": "Lulea", "country": "Sweden"},
        21: {"lat": 63.83, "long": 20.26, "location": "Umea", "country": "Sweden"},
        22: {"lat": 62.39, "long": 17.31, "location": "Sundsvall", "country": "Sweden"},
        23: {"lat": 60.67, "long": 17.14, "location": "Gavle", "country": "Sweden"},
        24: {"lat": 60.49, "long": 15.44, "location": "Borlaenge", "country": "Sweden"},
        25: {"lat": 59.86, "long": 17.65, "location": "Uppsala", "country": "Sweden"},
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 5.0, "fiber_length": 191.2018597141804},
        2.0: {"source": 1.0, "destination": 7.0, "fiber_length": 86.431679711237},
        3.0: {"source": 1.0, "destination": 8.0, "fiber_length": 99.19048442572924},
        4.0: {"source": 2.0, "destination": 3.0, "fiber_length": 187.9491363609758},
        5.0: {"source": 2.0, "destination": 7.0, "fiber_length": 137.6114807357079},
        6.0: {"source": 2.0, "destination": 8.0, "fiber_length": 137.8139272607867},
        7.0: {"source": 3.0, "destination": 24.0, "fiber_length": 204.3949421518425},
        8.0: {"source": 3.0, "destination": 4.0, "fiber_length": 146.6554391747205},
        9.0: {"source": 3.0, "destination": 18.0, "fiber_length": 1254.331218986416},
        10.0: {"source": 4.0, "destination": 8.0, "fiber_length": 211.1155177108748},
        11.0: {"source": 5.0, "destination": 11.0, "fiber_length": 163.023001573571},
        12.0: {"source": 6.0, "destination": 17.0, "fiber_length": 259.380880826334},
        13.0: {"source": 6.0, "destination": 9.0, "fiber_length": 167.1893618072713},
        14.0: {"source": 9.0, "destination": 10.0, "fiber_length": 162.9996187018616},
        15.0: {"source": 10.0, "destination": 11.0, "fiber_length": 245.3439784585527},
        16.0: {"source": 10.0, "destination": 14.0, "fiber_length": 145.0669710647389},
        17.0: {"source": 11.0, "destination": 16.0, "fiber_length": 24.44940720545842},
        18.0: {"source": 12.0, "destination": 13.0, "fiber_length": 109.2704753612958},
        19.0: {"source": 12.0, "destination": 16.0, "fiber_length": 128.5248585261045},
        20.0: {"source": 13.0, "destination": 14.0, "fiber_length": 124.707022398617},
        21.0: {"source": 15.0, "destination": 17.0, "fiber_length": 282.6542131096987},
        22.0: {"source": 17.0, "destination": 18.0, "fiber_length": 1155.194981620312},
        23.0: {"source": 17.0, "destination": 25.0, "fiber_length": 94.93331164885305},
        24.0: {"source": 19.0, "destination": 20.0, "fiber_length": 400.7487923966357},
        25.0: {"source": 20.0, "destination": 21.0, "fiber_length": 321.4322897642058},
        26.0: {"source": 21.0, "destination": 22.0, "fiber_length": 327.3660046923977},
        27.0: {"source": 22.0, "destination": 23.0, "fiber_length": 287.2008886170661},
        28.0: {"source": 23.0, "destination": 24.0, "fiber_length": 142.4751928955587},
        29.0: {"source": 23.0, "destination": 25.0, "fiber_length": 141.5353864152045},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
