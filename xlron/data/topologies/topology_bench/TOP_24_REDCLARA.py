def create_redclara_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': -35.119, 'long': -65.27, 'location': nan, 'country': nan},
    2: {'lat': -23.59, 'long': -46.57, 'location': 'sao paulo', 'country': nan},
    3: {'lat': -2.51, 'long': -44.26, 'location': 'Sao Luis', 'country': nan},
    4: {'lat': -32.76, 'long': -55.94, 'location': nan, 'country': nan},
    5: {'lat': -26.81, 'long': -70.5, 'location': nan, 'country': nan},
    6: {'lat': -0.94, 'long': -78.93, 'location': nan, 'country': nan},
    7: {'lat': 3.62, 'long': -73.48, 'location': nan, 'country': nan},
    8: {'lat': 8.51, 'long': -80.78, 'location': nan, 'country': nan},
    9: {'lat': 9.47, 'long': -85.61, 'location': nan, 'country': nan},
    10: {'lat': 15.73, 'long': -90.18, 'location': nan, 'country': nan},
    11: {'lat': 13.006, 'long': -83.59, 'location': nan, 'country': nan},
    12: {'lat': 14.81, 'long': -87.3, 'location': nan, 'country': nan},
    13: {'lat': 19.54, 'long': -99.28, 'location': nan, 'country': nan},
    14: {'lat': 25.88, 'long': -80.298, 'location': 'Miami', 'country': nan},
    }

    edge_attributes = {
    1.0: {'source': 8.0, 'destination': 9.0, 'fiber_length': 708.33},
    2.0: {'source': 8.0, 'destination': 14.0, 'fiber_length': 2333.9},
    3.0: {'source': 8.0, 'destination': 11.0, 'fiber_length': 1134.285},
    4.0: {'source': 8.0, 'destination': 12.0, 'fiber_length': 1500.0},
    5.0: {'source': 8.0, 'destination': 13.0, 'fiber_length': 3676.05},
    6.0: {'source': 8.0, 'destination': 10.0, 'fiber_length': 1733.475},
    7.0: {'source': 8.0, 'destination': 5.0, 'fiber_length': 6294.5875},
    8.0: {'source': 8.0, 'destination': 7.0, 'fiber_length': 1135.455},
    9.0: {'source': 14.0, 'destination': 6.0, 'fiber_length': 3844.7375},
    10.0: {'source': 14.0, 'destination': 3.0, 'fiber_length': 6227.9625},
    11.0: {'source': 7.0, 'destination': 6.0, 'fiber_length': 1248.87},
    12.0: {'source': 7.0, 'destination': 3.0, 'fiber_length': 4284.0375},
    13.0: {'source': 3.0, 'destination': 2.0, 'fiber_length': 2937.575},
    14.0: {'source': 3.0, 'destination': 2.0, 'fiber_length': 2937.575},
    15.0: {'source': 2.0, 'destination': 1.0, 'fiber_length': 2882.5125},
    16.0: {'source': 2.0, 'destination': 4.0, 'fiber_length': 1674.875},
    17.0: {'source': 2.0, 'destination': 5.0, 'fiber_length': 3434.8625},
    18.0: {'source': 5.0, 'destination': 1.0, 'fiber_length': 1149.45},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
