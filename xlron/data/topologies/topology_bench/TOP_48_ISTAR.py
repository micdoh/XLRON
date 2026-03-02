def create_istar_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 45.51, 'long': -73.59, 'location': 'Montreal', 'country': 'Canada'},
    2: {'lat': 46.81, 'long': -71.21, 'location': 'Quebec', 'country': 'Canada'},
    3: {'lat': 42.98, 'long': -81.23, 'location': 'London', 'country': 'Canada'},
    4: {'lat': 46.12, 'long': -64.8, 'location': 'Moncton', 'country': 'Canada'},
    5: {'lat': 44.65, 'long': -63.57, 'location': 'Halifax', 'country': 'Canada'},
    6: {'lat': 45.4, 'long': -71.9, 'location': 'Sherbrooke', 'country': 'Canada'},
    7: {'lat': 45.27, 'long': -66.07, 'location': 'Saint John', 'country': 'Canada'},
    8: {'lat': 46.14, 'long': -60.18, 'location': 'Sydney', 'country': 'Canada'},
    9: {'lat': 43.83, 'long': -66.12, 'location': 'Yarmouth', 'country': 'Canada'},
    10: {'lat': 48.43, 'long': -123.37, 'location': 'Victoria', 'country': 'Canada'},
    11: {'lat': 43.7, 'long': -79.42, 'location': 'Toronto', 'country': 'Canada'},
    12: {'lat': 43.23, 'long': -79.95, 'location': 'Ancaster', 'country': 'Canada'},
    13: {'lat': 49.25, 'long': -123.12, 'location': 'Vancouver', 'country': 'Canada'},
    14: {'lat': 51.05, 'long': -114.09, 'location': 'Calgary', 'country': 'Canada'},
    15: {'lat': 53.55, 'long': -113.47, 'location': 'Edmonton', 'country': 'Canada'},
    16: {'lat': 49.7, 'long': -112.82, 'location': 'Lethbridge', 'country': 'Canada'},
    17: {'lat': 49.88, 'long': -97.15, 'location': 'Winnipeg', 'country': 'Canada'},
    18: {'lat': 45.41, 'long': -75.7, 'location': 'Ottawa', 'country': 'Canada'},
    19: {'lat': 43.45, 'long': -80.48, 'location': 'Kitchener', 'country': 'Canada'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 350.1389724135474},
    2.0: {'source': 1.0, 'destination': 4.0, 'fiber_length': 1026.361517307967},
    3.0: {'source': 1.0, 'destination': 11.0, 'fiber_length': 755.0357670265992},
    4.0: {'source': 1.0, 'destination': 6.0, 'fiber_length': 198.5751791544305},
    5.0: {'source': 1.0, 'destination': 18.0, 'fiber_length': 247.4032001412376},
    6.0: {'source': 3.0, 'destination': 11.0, 'fiber_length': 250.2531794080375},
    7.0: {'source': 4.0, 'destination': 5.0, 'fiber_length': 284.3799353856231},
    8.0: {'source': 4.0, 'destination': 7.0, 'fiber_length': 204.9111545304429},
    9.0: {'source': 5.0, 'destination': 8.0, 'fiber_length': 468.3506272246442},
    10.0: {'source': 5.0, 'destination': 9.0, 'fiber_length': 333.9767249411377},
    11.0: {'source': 10.0, 'destination': 13.0, 'fiber_length': 139.4958300540092},
    12.0: {'source': 11.0, 'destination': 12.0, 'fiber_length': 101.3005820479409},
    13.0: {'source': 11.0, 'destination': 18.0, 'fiber_length': 526.0580396293944},
    14.0: {'source': 11.0, 'destination': 19.0, 'fiber_length': 134.7019718699122},
    15.0: {'source': 13.0, 'destination': 14.0, 'fiber_length': 1009.95690485004},
    16.0: {'source': 13.0, 'destination': 18.0, 'fiber_length': 4424.968959600237},
    17.0: {'source': 14.0, 'destination': 15.0, 'fiber_length': 421.7445112294374},
    18.0: {'source': 14.0, 'destination': 16.0, 'fiber_length': 262.5774509897452},
    19.0: {'source': 17.0, 'destination': 18.0, 'fiber_length': 2094.139590166251},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
