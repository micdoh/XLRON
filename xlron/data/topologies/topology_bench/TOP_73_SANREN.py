def create_sanren_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {"lat": -26.2, "long": 28.04, "location": "Johannesburg", "country": "South Africa"},
        2: {"lat": -25.74, "long": 28.19, "location": "Pretoria", "country": "South Africa"},
        3: {"lat": -29.85, "long": 31.02, "location": "Durban", "country": "South Africa"},
        4: {"lat": -29.13, "long": 26.2, "location": "Bloemfontein", "country": "South Africa"},
        5: {"lat": -33.02, "long": 27.91, "location": "East London", "country": "South Africa"},
        6: {"lat": -33.97, "long": 25.58, "location": "Port Elizabeth", "country": "South Africa"},
        7: {"lat": -33.92, "long": 18.42, "location": "Cape Town", "country": "South Africa"},
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 2.0, "fiber_length": 79.95347697325593},
        2.0: {"source": 1.0, "destination": 4.0, "fiber_length": 559.1786021926544},
        3.0: {"source": 2.0, "destination": 3.0, "fiber_length": 802.5869299607084},
        4.0: {"source": 3.0, "destination": 5.0, "fiber_length": 689.4391493981935},
        5.0: {"source": 4.0, "destination": 7.0, "fiber_length": 1363.739306447085},
        6.0: {"source": 5.0, "destination": 6.0, "fiber_length": 360.7364115689852},
        7.0: {"source": 6.0, "destination": 7.0, "fiber_length": 990.5388718653302},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
