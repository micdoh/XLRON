def create_ernet_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {"lat": 18.52, "long": 73.86, "location": "Pune", "country": "India"},
        2: {"lat": 22.72, "long": 75.83, "location": "Indore", "country": "India"},
        3: {"lat": 8.48, "long": 76.92, "location": "Thiruvananthapuram", "country": "India"},
        4: {"lat": 19.01, "long": 72.85, "location": "Mumbai", "country": "India"},
        5: {"lat": 23.03, "long": 72.62, "location": "Ahmedabad", "country": "India"},
        6: {"lat": 26.92, "long": 75.82, "location": "Jaipur", "country": "India"},
        7: {"lat": 13.09, "long": 80.28, "location": "George Town", "country": "India"},
        8: {"lat": 12.98, "long": 77.6, "location": "Bangalore", "country": "India"},
        9: {"lat": 28.64, "long": 77.22, "location": "New Delhi", "country": "India"},
        10: {"lat": 26.19, "long": 91.75, "location": "Guwahati", "country": "India"},
        11: {"lat": 29.45, "long": 75.68, "location": "Gorakhpur", "country": "India"},
        12: {"lat": 26.47, "long": 80.35, "location": "Kanpur", "country": "India"},
        13: {"lat": 25.45, "long": 81.85, "location": "Allahabad", "country": "India"},
        14: {"lat": 22.57, "long": 88.37, "location": "Kolkata", "country": "India"},
        15: {"lat": 20.23, "long": 85.83, "location": "Bhubaneshwar", "country": "India"},
        16: {"lat": 17.38, "long": 78.47, "location": "Hyderabad", "country": "India"},
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 4.0, "fiber_length": 179.224401832357},
        2.0: {"source": 1.0, "destination": 8.0, "fiber_length": 1101.786486959481},
        3.0: {"source": 2.0, "destination": 4.0, "fiber_length": 773.6337951082263},
        4.0: {"source": 3.0, "destination": 8.0, "fiber_length": 758.7883770649184},
        5.0: {"source": 4.0, "destination": 9.0, "fiber_length": 1500.0},
        6.0: {"source": 5.0, "destination": 9.0, "fiber_length": 1162.635332833133},
        7.0: {"source": 6.0, "destination": 9.0, "fiber_length": 353.5213518954667},
        8.0: {"source": 7.0, "destination": 8.0, "fiber_length": 435.869644797688},
        9.0: {"source": 7.0, "destination": 16.0, "fiber_length": 772.5246083159809},
        10.0: {"source": 8.0, "destination": 9.0, "fiber_length": 2177.195551264161},
        11.0: {"source": 8.0, "destination": 16.0, "fiber_length": 747.1215199807971},
        12.0: {"source": 9.0, "destination": 12.0, "fiber_length": 587.5069712378195},
        13.0: {"source": 9.0, "destination": 14.0, "fiber_length": 1631.198779249713},
        14.0: {"source": 10.0, "destination": 14.0, "fiber_length": 792.5113756660074},
        15.0: {"source": 11.0, "destination": 12.0, "fiber_length": 848.6111993950832},
        16.0: {"source": 12.0, "destination": 13.0, "fiber_length": 282.0298130830362},
        17.0: {"source": 14.0, "destination": 15.0, "fiber_length": 554.8701312485068},
        18.0: {"source": 14.0, "destination": 16.0, "fiber_length": 1500.0},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
