def create_arnes_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1.0: {'lat': 46.55, 'long': 15.633333333333, 'location': nan, 'country': nan},
    2.0: {'lat': 46.59004243, 'long': 15.017967473, 'location': nan, 'country': nan},
    3.0: {'lat': 46.366666666667, 'long': 15.116666666667, 'location': nan, 'country': nan},
    4.0: {'lat': 46.22909194, 'long': 15.264128741, 'location': nan, 'country': nan},
    5.0: {'lat': 46.394136552, 'long': 15.570726245, 'location': nan, 'country': nan},
    6.0: {'lat': 46.156303, 'long': 15.238617, 'location': nan, 'country': nan},
    7.0: {'lat': 46.15, 'long': 15.05, 'location': nan, 'country': nan},
    8.0: {'lat': 45.846325, 'long': 15.424869, 'location': nan, 'country': nan},
    9.0: {'lat': 45.798055555556, 'long': 15.162777777778, 'location': nan, 'country': nan},
    10.0: {'lat': 45.642961111111, 'long': 14.859383333333, 'location': nan, 'country': nan},
    11.0: {'lat': 46.0513888888889, 'long': 14.5061111111111, 'location': nan, 'country': nan},
    12.0: {'lat': 46.23887, 'long': 14.35561, 'location': nan, 'country': nan},
    13.0: {'lat': 46.366666666667, 'long': 14.116666666667, 'location': nan, 'country': nan},
    14.0: {'lat': 46.183, 'long': 13.733, 'location': nan, 'country': nan},
    15.0: {'lat': 45.966666666667, 'long': 13.65, 'location': nan, 'country': nan},
    16.0: {'lat': 45.888356136, 'long': 13.905237912, 'location': nan, 'country': nan},
    17.0: {'lat': 45.547542728, 'long': 13.7306571, 'location': nan, 'country': nan},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 71.52},
    2.0: {'source': 2.0, 'destination': 3.0, 'fiber_length': 39.825},
    3.0: {'source': 3.0, 'destination': 4.0, 'fiber_length': 27.48},
    4.0: {'source': 4.0, 'destination': 5.0, 'fiber_length': 44.94},
    5.0: {'source': 5.0, 'destination': 1.0, 'fiber_length': 28.875},
    6.0: {'source': 4.0, 'destination': 6.0, 'fiber_length': 12.96},
    7.0: {'source': 6.0, 'destination': 7.0, 'fiber_length': 20.97},
    8.0: {'source': 7.0, 'destination': 11.0, 'fiber_length': 65.685},
    9.0: {'source': 11.0, 'destination': 4.0, 'fiber_length': 92.22},
    10.0: {'source': 5.0, 'destination': 8.0, 'fiber_length': 92.80499999999999},
    11.0: {'source': 8.0, 'destination': 9.0, 'fiber_length': 29.94},
    12.0: {'source': 9.0, 'destination': 10.0, 'fiber_length': 45.405},
    13.0: {'source': 10.0, 'destination': 11.0, 'fiber_length': 79.035},
    14.0: {'source': 11.0, 'destination': 17.0, 'fiber_length': 123.135},
    15.0: {'source': 17.0, 'destination': 16.0, 'fiber_length': 60.33},
    16.0: {'source': 16.0, 'destination': 15.0, 'fiber_length': 32.04},
    17.0: {'source': 15.0, 'destination': 14.0, 'fiber_length': 38.805},
    18.0: {'source': 14.0, 'destination': 13.0, 'fiber_length': 53.325},
    19.0: {'source': 13.0, 'destination': 12.0, 'fiber_length': 35.09999999999999},
    20.0: {'source': 12.0, 'destination': 11.0, 'fiber_length': 36.705},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
