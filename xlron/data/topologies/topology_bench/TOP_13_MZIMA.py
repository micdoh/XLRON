def create_mzima_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 47.6038321, 'long': -122.330062, 'location': 'Seattle', 'country': nan},
    2: {'lat': 37.7792588, 'long': -122.4193286, 'location': 'San Francisco', 'country': nan},
    3: {'lat': 40.7596198, 'long': -111.886797, 'location': 'Salt Lake City', 'country': nan},
    4: {'lat': 36.1672559, 'long': -115.148516, 'location': 'Las Vegas', 'country': nan},
    5: {'lat': 34.0536909, 'long': -118.242766, 'location': 'Los Angeles', 'country': nan},
    6: {'lat': 33.4484367, 'long': -112.074141, 'location': 'Phoenix', 'country': nan},
    7: {'lat': 29.7589382, 'long': -95.3676974, 'location': 'Houston', 'country': nan},
    8: {'lat': 32.7762719, 'long': -96.7968559, 'location': 'Dallas', 'country': nan},
    9: {'lat': 39.100105, 'long': -94.5781416, 'location': 'Kansas City', 'country': nan},
    10: {'lat': 25.7741728, 'long': -80.19362, 'location': 'Miami', 'country': nan},
    11: {'lat': 33.7489924, 'long': -84.3902644, 'location': 'Atlanta', 'country': nan},
    12: {'lat': 41.8755616, 'long': -87.6244212, 'location': 'Chicago', 'country': nan},
    13: {'lat': 39.0300191, 'long': -77.4696464655765, 'location': 'Ash Burn', 'country': nan},
    14: {'lat': 40.7127281, 'long': -74.0060152, 'location': 'New York', 'country': nan},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 3.0, 'fiber_length': 1500.0},
    2.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 1500.0},
    3.0: {'source': 2.0, 'destination': 3.0, 'fiber_length': 1446.885},
    4.0: {'source': 2.0, 'destination': 5.0, 'fiber_length': 839.0999999999999},
    5.0: {'source': 3.0, 'destination': 4.0, 'fiber_length': 876.2850000000001},
    6.0: {'source': 3.0, 'destination': 9.0, 'fiber_length': 1856.125},
    7.0: {'source': 4.0, 'destination': 5.0, 'fiber_length': 549.96},
    8.0: {'source': 5.0, 'destination': 6.0, 'fiber_length': 861.27},
    9.0: {'source': 6.0, 'destination': 7.0, 'fiber_length': 2040.5375},
    10.0: {'source': 7.0, 'destination': 8.0, 'fiber_length': 542.925},
    11.0: {'source': 7.0, 'destination': 11.0, 'fiber_length': 1500.0},
    12.0: {'source': 7.0, 'destination': 10.0, 'fiber_length': 1944.9},
    13.0: {'source': 8.0, 'destination': 9.0, 'fiber_length': 1096.395},
    14.0: {'source': 9.0, 'destination': 12.0, 'fiber_length': 995.7450000000001},
    15.0: {'source': 10.0, 'destination': 11.0, 'fiber_length': 1461.945},
    16.0: {'source': 11.0, 'destination': 13.0, 'fiber_length': 1279.59},
    17.0: {'source': 12.0, 'destination': 14.0, 'fiber_length': 1500.0},
    18.0: {'source': 13.0, 'destination': 14.0, 'fiber_length': 524.6850000000001},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
