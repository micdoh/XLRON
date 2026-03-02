def create_canarie19_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 49.2823479514009, 'long': -123.125966385064, 'location': 'Vancouver', 'country': 'Canada'},
    2: {'lat': 47.6049578648434, 'long': -122.326211588261, 'location': 'Seattle', 'country': 'USA'},
    3: {'lat': 50.6767836790158, 'long': -120.319139574281, 'location': 'Kamloops', 'country': 'Canada'},
    4: {'lat': 53.5473437425427, 'long': -113.514816959252, 'location': 'Edmonton', 'country': 'Canada'},
    5: {'lat': 51.0457750458443, 'long': -114.081714898762, 'location': 'Calgary', 'country': 'Canada'},
    6: {'lat': 52.1583440859846, 'long': -106.666475088772, 'location': 'Saskatoon', 'country': 'Canada'},
    7: {'lat': 50.4503129242101, 'long': -104.612048365926, 'location': 'Regina', 'country': 'Canada'},
    8: {'lat': 49.8970298284167, 'long': -97.1233236908463, 'location': 'Wimipeg', 'country': 'Canada'},
    9: {'lat': 48.3820754154255, 'long': -89.2433348006063, 'location': 'Thunderbay', 'country': 'Canada'},
    10: {'lat': 41.8781096951375, 'long': -87.6488908438788, 'location': 'Chicago', 'country': 'USA'},
    11: {'lat': 42.3311875808133, 'long': -83.0530290467165, 'location': 'Detroit', 'country': 'USA'},
    12: {'lat': 43.6530759154005, 'long': -79.3663957638143, 'location': 'Toronto', 'country': 'Canada'},
    13: {'lat': 45.4989357041362, 'long': -73.562881651828, 'location': 'Montreal', 'country': 'Canada'},
    14: {'lat': 45.419588393874, 'long': -75.6973013088981, 'location': 'Ottawa', 'country': 'Canada'},
    15: {'lat': 45.9635726858031, 'long': -66.6463414238811, 'location': 'Fredericton', 'country': 'Canada'},
    16: {'lat': 44.6504201124915, 'long': -63.6036577586685, 'location': 'Halifax', 'country': 'Canada'},
    17: {'lat': 42.3594938804217, 'long': -71.0502809403789, 'location': 'Boston', 'country': 'USA'},
    18: {'lat': 40.7098639288127, 'long': -74.0099045805336, 'location': 'New York', 'country': 'USA'},
    19: {'lat': 48.4283098463054, 'long': -123.365403310915, 'location': 'Victoria', 'country': 'Canada'},
    }

    edge_attributes = {
    1.0: {'source': 19.0, 'destination': 2.0, 'fiber_length': 179.565},
    2.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 289.68},
    3.0: {'source': 19.0, 'destination': 1.0, 'fiber_length': 141.57},
    4.0: {'source': 1.0, 'destination': 3.0, 'fiber_length': 379.83},
    5.0: {'source': 3.0, 'destination': 5.0, 'fiber_length': 664.02},
    6.0: {'source': 5.0, 'destination': 4.0, 'fiber_length': 421.0649999999999},
    7.0: {'source': 5.0, 'destination': 7.0, 'fiber_length': 1000.56},
    8.0: {'source': 5.0, 'destination': 8.0, 'fiber_length': 1501.9125},
    9.0: {'source': 5.0, 'destination': 10.0, 'fiber_length': 2812.6125},
    10.0: {'source': 2.0, 'destination': 10.0, 'fiber_length': 3486.5375},
    11.0: {'source': 4.0, 'destination': 6.0, 'fiber_length': 727.155},
    12.0: {'source': 6.0, 'destination': 8.0, 'fiber_length': 1065.585},
    13.0: {'source': 8.0, 'destination': 9.0, 'fiber_length': 1500.0},
    14.0: {'source': 8.0, 'destination': 12.0, 'fiber_length': 1893.3125},
    15.0: {'source': 9.0, 'destination': 12.0, 'fiber_length': 731.91},
    16.0: {'source': 7.0, 'destination': 8.0, 'fiber_length': 803.9100000000001},
    17.0: {'source': 10.0, 'destination': 11.0, 'fiber_length': 571.5},
    18.0: {'source': 11.0, 'destination': 12.0, 'fiber_length': 498.225},
    19.0: {'source': 12.0, 'destination': 18.0, 'fiber_length': 825.765},
    20.0: {'source': 12.0, 'destination': 14.0, 'fiber_length': 528.84},
    21.0: {'source': 14.0, 'destination': 13.0, 'fiber_length': 248.415},
    22.0: {'source': 18.0, 'destination': 16.0, 'fiber_length': 1434.945},
    23.0: {'source': 18.0, 'destination': 17.0, 'fiber_length': 458.55},
    24.0: {'source': 17.0, 'destination': 13.0, 'fiber_length': 605.325},
    25.0: {'source': 13.0, 'destination': 15.0, 'fiber_length': 809.79},
    26.0: {'source': 15.0, 'destination': 16.0, 'fiber_length': 420.3149999999999},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
