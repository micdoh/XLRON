def create_bbnplanet_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 32.3, 'long': -90.18, 'location': 'Jackson', 'country': 'United States of America (the)'},
    2: {'lat': 33.75, 'long': -84.39, 'location': 'Atlanta', 'country': 'United States of America (the)'},
    3: {'lat': 30.27, 'long': -97.74, 'location': 'Austin', 'country': 'United States of America (the)'},
    4: {'lat': 29.76, 'long': -95.36, 'location': 'Houston', 'country': 'United States of America (the)'},
    5: {'lat': 42.33, 'long': -83.05, 'location': 'Detroit', 'country': 'United States of America (the)'},
    6: {'lat': 39.96, 'long': -83.0, 'location': 'Columbus', 'country': 'United States of America (the)'},
    7: {'lat': 39.16, 'long': -84.46, 'location': 'Cincinnati', 'country': 'United States of America (the)'},
    8: {'lat': 41.5, 'long': -81.7, 'location': 'Cleveland', 'country': 'United States of America (the)'},
    9: {'lat': 41.85, 'long': -87.65, 'location': 'Chicago', 'country': 'United States of America (the)'},
    10: {'lat': 44.98, 'long': -93.26, 'location': 'Minneapolis', 'country': 'United States of America (the)'},
    11: {'lat': 39.95, 'long': -75.16, 'location': 'Philadelphia', 'country': 'United States of America (the)'},
    12: {'lat': 40.71, 'long': -74.01, 'location': 'New York City', 'country': 'United States of America (the)'},
    13: {'lat': 42.38, 'long': -71.11, 'location': 'Cambridge', 'country': 'United States of America (the)'},
    14: {'lat': 42.36, 'long': -71.06, 'location': 'Boston', 'country': 'United States of America (the)'},
    15: {'lat': 37.55, 'long': -77.46, 'location': 'Richmond', 'country': 'United States of America (the)'},
    16: {'lat': 38.9, 'long': -77.04, 'location': 'Washington, D.C.', 'country': 'United States of America (the)'},
    17: {'lat': 39.29, 'long': -76.61, 'location': 'Baltimore', 'country': 'United States of America (the)'},
    18: {'lat': 39.74, 'long': -104.98, 'location': 'Denver', 'country': 'United States of America (the)'},
    19: {'lat': 32.78, 'long': -96.81, 'location': 'Dallas', 'country': 'United States of America (the)'},
    20: {'lat': 38.58, 'long': -121.49, 'location': 'Sacramento', 'country': 'United States of America (the)'},
    21: {'lat': 37.8, 'long': -122.27, 'location': 'Oakland', 'country': 'United States of America (the)'},
    22: {'lat': 37.77, 'long': -122.42, 'location': 'San Francisco', 'country': 'United States of America (the)'},
    23: {'lat': 37.44, 'long': -122.14, 'location': 'Palo Alto', 'country': 'United States of America (the)'},
    24: {'lat': 37.34, 'long': -121.89, 'location': 'San Jose', 'country': 'United States of America (the)'},
    25: {'lat': 34.05, 'long': -118.24, 'location': 'Los Angeles', 'country': 'United States of America (the)'},
    26: {'lat': 32.72, 'long': -117.16, 'location': 'San Diego', 'country': 'United States of America (the)'},
    27: {'lat': 33.79, 'long': -117.85, 'location': 'Orange', 'country': 'United States of America (the)'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 844.9003647984742},
    2.0: {'source': 2.0, 'destination': 19.0, 'fiber_length': 1500.0},
    3.0: {'source': 2.0, 'destination': 16.0, 'fiber_length': 1308.123444881084},
    4.0: {'source': 3.0, 'destination': 19.0, 'fiber_length': 439.0260901564217},
    5.0: {'source': 4.0, 'destination': 19.0, 'fiber_length': 544.4600499700148},
    6.0: {'source': 5.0, 'destination': 8.0, 'fiber_length': 217.3432784660523},
    7.0: {'source': 6.0, 'destination': 8.0, 'fiber_length': 304.9098215971796},
    8.0: {'source': 7.0, 'destination': 8.0, 'fiber_length': 524.8062575383049},
    9.0: {'source': 8.0, 'destination': 9.0, 'fiber_length': 743.4071200946746},
    10.0: {'source': 8.0, 'destination': 12.0, 'fiber_length': 975.077114932617},
    11.0: {'source': 9.0, 'destination': 18.0, 'fiber_length': 1843.716434385458},
    12.0: {'source': 9.0, 'destination': 14.0, 'fiber_length': 1709.543659108899},
    13.0: {'source': 9.0, 'destination': 10.0, 'fiber_length': 856.6967858075002},
    14.0: {'source': 11.0, 'destination': 12.0, 'fiber_length': 193.5165266446945},
    15.0: {'source': 12.0, 'destination': 13.0, 'fiber_length': 456.7262459554615},
    16.0: {'source': 12.0, 'destination': 16.0, 'fiber_length': 491.7553096743694},
    17.0: {'source': 15.0, 'destination': 16.0, 'fiber_length': 231.7964268611971},
    18.0: {'source': 16.0, 'destination': 17.0, 'fiber_length': 85.61340860642284},
    19.0: {'source': 16.0, 'destination': 24.0, 'fiber_length': 4859.280098637835},
    20.0: {'source': 18.0, 'destination': 21.0, 'fiber_length': 1889.96295604635},
    21.0: {'source': 19.0, 'destination': 25.0, 'fiber_length': 2487.99326220719},
    22.0: {'source': 20.0, 'destination': 21.0, 'fiber_length': 165.4707141559489},
    23.0: {'source': 21.0, 'destination': 22.0, 'fiber_length': 20.39609906326637},
    24.0: {'source': 21.0, 'destination': 24.0, 'fiber_length': 91.70772793515846},
    25.0: {'source': 23.0, 'destination': 24.0, 'fiber_length': 37.0916753144256},
    26.0: {'source': 24.0, 'destination': 25.0, 'fiber_length': 738.4958532321912},
    27.0: {'source': 25.0, 'destination': 26.0, 'fiber_length': 268.0139459776118},
    28.0: {'source': 25.0, 'destination': 27.0, 'fiber_length': 69.24095902905727},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
