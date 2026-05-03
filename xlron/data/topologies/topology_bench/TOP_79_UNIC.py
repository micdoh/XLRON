def create_unic_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {"lat": 55.4, "long": 10.39, "location": "Odense", "country": "Denmark"},
        2: {"lat": 55.31, "long": 10.79, "location": "Nyborg", "country": "Denmark"},
        3: {"lat": 55.71, "long": 9.54, "location": "Vejle", "country": "Denmark"},
        4: {"lat": 55.49, "long": 9.47, "location": "Kolding", "country": "Denmark"},
        5: {"lat": 55.72, "long": 11.72, "location": "Holbaek", "country": "Denmark"},
        6: {"lat": 55.77, "long": 12.5, "location": "Kongens Lyngby", "country": "Denmark"},
        7: {"lat": 55.4, "long": 11.35, "location": "Slagelse", "country": "Denmark"},
        8: {"lat": 55.31, "long": 11.55, "location": "Fuglebjerg", "country": "Denmark"},
        9: {"lat": 55.63, "long": 12.6, "location": "Tarnby", "country": "Denmark"},
        10: {"lat": 56.14, "long": 8.97, "location": "Herning", "country": "Denmark"},
        11: {"lat": 57.05, "long": 9.92, "location": "Aalborg", "country": "Denmark"},
        12: {"lat": 56.77, "long": 9.34, "location": "Farso", "country": "Denmark"},
        13: {"lat": 56.64, "long": 9.79, "location": "Hobro", "country": "Denmark"},
        14: {"lat": 56.17, "long": 9.55, "location": "Silkeborg", "country": "Denmark"},
        15: {"lat": 56.16, "long": 10.21, "location": "Arhus", "country": "Denmark"},
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 2.0, "fiber_length": 40.7904531096874},
        2.0: {"source": 1.0, "destination": 3.0, "fiber_length": 95.41272464370458},
        3.0: {"source": 1.0, "destination": 4.0, "fiber_length": 88.32017008371076},
        4.0: {"source": 2.0, "destination": 7.0, "fiber_length": 55.18001675264059},
        5.0: {"source": 2.0, "destination": 8.0, "fiber_length": 72.14458617985376},
        6.0: {"source": 3.0, "destination": 10.0, "fiber_length": 89.33682810863385},
        7.0: {"source": 3.0, "destination": 4.0, "fiber_length": 37.2824852980827},
        8.0: {"source": 3.0, "destination": 15.0, "fiber_length": 97.73171142212857},
        9.0: {"source": 5.0, "destination": 6.0, "fiber_length": 73.70215853435309},
        10.0: {"source": 5.0, "destination": 7.0, "fiber_length": 63.77164435587142},
        11.0: {"source": 6.0, "destination": 9.0, "fiber_length": 25.1716208207202},
        12.0: {"source": 8.0, "destination": 9.0, "fiber_length": 112.7083722562153},
        13.0: {"source": 10.0, "destination": 14.0, "fiber_length": 54.11061870398387},
        14.0: {"source": 11.0, "destination": 12.0, "fiber_length": 70.5016566089086},
        15.0: {"source": 11.0, "destination": 13.0, "fiber_length": 69.40542136891197},
        16.0: {"source": 12.0, "destination": 14.0, "fiber_length": 101.9283927314822},
        17.0: {"source": 13.0, "destination": 15.0, "fiber_length": 88.95188344508486},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
