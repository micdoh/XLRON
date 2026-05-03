def create_noel_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
        1: {
            "lat": 47.66,
            "long": -117.43,
            "location": "Spokane",
            "country": "United States of America (the)",
        },
        2: {
            "lat": 47.68,
            "long": -116.78,
            "location": "Coeur d'Alene",
            "country": "United States of America (the)",
        },
        3: {
            "lat": 47.42,
            "long": -120.31,
            "location": "Wenatchee",
            "country": "United States of America (the)",
        },
        4: {
            "lat": 46.83,
            "long": -119.18,
            "location": "Othello",
            "country": "United States of America (the)",
        },
        5: {
            "lat": 46.06,
            "long": -118.34,
            "location": "Walla Walla",
            "country": "United States of America (the)",
        },
        6: {
            "lat": 46.24,
            "long": -119.1,
            "location": "Pasco",
            "country": "United States of America (the)",
        },
        7: {
            "lat": 46.73,
            "long": -117.18,
            "location": "Pullman",
            "country": "United States of America (the)",
        },
        8: {
            "lat": 46.42,
            "long": -117.02,
            "location": "Lewiston",
            "country": "United States of America (the)",
        },
        9: {
            "lat": 46.21,
            "long": -119.14,
            "location": "Kennewick",
            "country": "United States of America (the)",
        },
        10: {
            "lat": 46.6,
            "long": -120.51,
            "location": "Yakima",
            "country": "United States of America (the)",
        },
        11: {
            "lat": 47.0,
            "long": -120.55,
            "location": "Ellensburg",
            "country": "United States of America (the)",
        },
        12: {"lat": 49.25, "long": -123.12, "location": "Vancouver", "country": "Canada"},
        13: {
            "lat": 48.76,
            "long": -122.49,
            "location": "Bellingham",
            "country": "United States of America (the)",
        },
        14: {
            "lat": 48.42,
            "long": -122.33,
            "location": "Mount Vernon",
            "country": "United States of America (the)",
        },
        15: {
            "lat": 47.98,
            "long": -122.2,
            "location": "Everett",
            "country": "United States of America (the)",
        },
        16: {
            "lat": 47.61,
            "long": -122.34,
            "location": "Seattle",
            "country": "United States of America (the)",
        },
        17: {
            "lat": 47.25,
            "long": -122.44,
            "location": "Tacoma",
            "country": "United States of America (the)",
        },
        18: {
            "lat": 47.04,
            "long": -122.9,
            "location": "Olympia",
            "country": "United States of America (the)",
        },
        19: {
            "lat": 45.52,
            "long": -122.68,
            "location": "Portland",
            "country": "United States of America (the)",
        },
    }

    edge_attributes = {
        1.0: {"source": 1.0, "destination": 2.0, "fiber_length": 73.08261974195123},
        2.0: {"source": 1.0, "destination": 3.0, "fiber_length": 326.7224503254238},
        3.0: {"source": 1.0, "destination": 4.0, "fiber_length": 241.7112063035856},
        4.0: {"source": 1.0, "destination": 7.0, "fiber_length": 157.6832261384525},
        5.0: {"source": 3.0, "destination": 11.0, "fiber_length": 75.14546248417876},
        6.0: {"source": 3.0, "destination": 16.0, "fiber_length": 230.8602144919846},
        7.0: {"source": 4.0, "destination": 10.0, "fiber_length": 156.8566505681249},
        8.0: {"source": 5.0, "destination": 9.0, "fiber_length": 95.78907487116305},
        9.0: {"source": 5.0, "destination": 6.0, "fiber_length": 92.80716702310214},
        10.0: {"source": 6.0, "destination": 9.0, "fiber_length": 6.80750400672969},
        11.0: {"source": 6.0, "destination": 10.0, "fiber_length": 172.8824903246685},
        12.0: {"source": 7.0, "destination": 8.0, "fiber_length": 54.86341642985107},
        13.0: {"source": 9.0, "destination": 10.0, "fiber_length": 170.4636330891594},
        14.0: {"source": 9.0, "destination": 19.0, "fiber_length": 426.9179729563697},
        15.0: {"source": 10.0, "destination": 11.0, "fiber_length": 66.87308990199159},
        16.0: {"source": 10.0, "destination": 16.0, "fiber_length": 267.4600221751385},
        17.0: {"source": 12.0, "destination": 13.0, "fiber_length": 106.9151766804321},
        18.0: {"source": 12.0, "destination": 16.0, "fiber_length": 286.8332012043319},
        19.0: {"source": 13.0, "destination": 14.0, "fiber_length": 59.39308239786348},
        20.0: {"source": 14.0, "destination": 15.0, "fiber_length": 74.79814172574787},
        21.0: {"source": 15.0, "destination": 16.0, "fiber_length": 63.67565608224866},
        22.0: {"source": 16.0, "destination": 17.0, "fiber_length": 61.09619682834861},
        23.0: {"source": 16.0, "destination": 19.0, "fiber_length": 350.7687027868379},
        24.0: {"source": 17.0, "destination": 18.0, "fiber_length": 62.84884443637591},
        25.0: {"source": 18.0, "destination": 19.0, "fiber_length": 254.7894079178984},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge["source"], edge["destination"], weight=edge["fiber_length"])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
