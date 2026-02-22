def create_loni_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 32.5135356, 'long': -93.7477839, 'location': 'Shreveport Louisiana', 'country': nan},
    2: {'lat': 32.5326514, 'long': -93.5040627, 'location': 'Haughton Louisiana', 'country': nan},
    3: {'lat': 32.5297498, 'long': -92.6386604, 'location': 'Ruston Louisiana', 'country': nan},
    4: {'lat': 32.5102427, 'long': -92.1032411, 'location': 'Monroe Louisiana', 'country': nan},
    5: {'lat': 32.4084539, 'long': -91.1869112, 'location': 'Tallulah Louisiana', 'country': nan},
    6: {'lat': 32.3301465, 'long': -90.6056548, 'location': 'Edwards Mississippi', 'country': nan},
    7: {'lat': 32.2998686, 'long': -90.1830408, 'location': 'Jackson Mississippi', 'country': nan},
    8: {'lat': 31.9622006, 'long': -89.8702079, 'location': 'Mendenhall Mississippi', 'country': nan},
    9: {'lat': 31.5623892, 'long': -89.4975669, 'location': 'Seminary Mississipi', 'country': nan},
    10: {'lat': 31.1160119, 'long': -90.1420331, 'location': 'Tylertown Mississipi', 'country': nan},
    11: {'lat': 30.829635, 'long': -90.6677355, 'location': 'Greensburg Louisiana', 'country': nan},
    12: {'lat': 32.2932692, 'long': -92.5628442, 'location': 'Jackson Louisiana', 'country': nan},
    13: {'lat': 30.4057618499999, 'long': -91.1859745026568, 'location': 'Louisiana State University', 'country': nan},
    14: {'lat': 30.5043583, 'long': -90.4611995, 'location': 'Hammond Louisiana', 'country': nan},
    15: {'lat': 30.0715993, 'long': -90.4663589046885, 'location': 'Laplace Louisiana', 'country': nan},
    16: {'lat': 29.956837, 'long': -90.0831349957948, 'location': 'Louisiana State University Health Sciences Center New Orleans', 'country': nan},
    17: {'lat': 30.0290509, 'long': -90.0650149, 'location': 'The University of New Orleans', 'country': nan},
    18: {'lat': 29.94121955, 'long': -90.1201008917918, 'location': 'Tulane University', 'country': nan},
    19: {'lat': 29.7386285, 'long': -90.81026625563359, 'location': 'Schriever Louisiana', 'country': nan},
    20: {'lat': 32.1254252, 'long': -91.6778053, 'location': 'Franklin Louisiana', 'country': nan},
    21: {'lat': 30.2105847574037, 'long': -92.0294380456438, 'location': 'University of Louisiana at Lafayette East University', 'country': nan},
    22: {'lat': 30.5601953, 'long': -91.9540055, 'location': 'Port Barre Louisiana', 'country': nan},
    23: {'lat': 30.4024175, 'long': -91.5073329, 'location': 'Ramah Louisiana', 'country': nan},
    24: {'lat': 30.2351806, 'long': -92.7478897, 'location': 'Roanoke Louisiana', 'country': nan},
    25: {'lat': 30.2305095, 'long': -93.2169807, 'location': 'Lake Charles Louisiana', 'country': nan},
    26: {'lat': 30.2140928, 'long': -92.3745761, 'location': 'Crowley Louisiana', 'country': nan},
    27: {'lat': 29.8693722, 'long': -91.6653928, 'location': 'Landry Louisiana', 'country': nan},
    28: {'lat': 31.3119463, 'long': -92.4453558, 'location': 'Alexandria Louisiana', 'country': nan},
    29: {'lat': 31.5340592, 'long': -92.9476592, 'location': 'Derry Louisiana', 'country': nan},
    30: {'lat': 32.0148834, 'long': -93.3421165, 'location': 'Coushatta Louisiana', 'country': nan},
    31: {'lat': 30.52581565, 'long': -91.1949460453526, 'location': 'Southern University and A&M College', 'country': nan},
    32: {'lat': 30.394061, 'long': -91.105683219937, 'location': 'Louisiana Public Broadcasting', 'country': nan},
    }

    edge_attributes = {
    1: {'source': 1, 'destination': 2, 'fiber_length': 22.9491859356208},
    2: {'source': 1, 'destination': 30, 'fiber_length': 67.3003276131565},
    3: {'source': 2, 'destination': 3, 'fiber_length': 81.1305265075258},
    4: {'source': 4, 'destination': 5, 'fiber_length': 86.7146875444426},
    5: {'source': 5, 'destination': 6, 'fiber_length': 55.279873058304},
    6: {'source': 6, 'destination': 7, 'fiber_length': 39.8568950272095},
    7: {'source': 7, 'destination': 8, 'fiber_length': 47.7233532314679},
    8: {'source': 8, 'destination': 9, 'fiber_length': 56.7238765225753},
    9: {'source': 9, 'destination': 10, 'fiber_length': 78.8022113602059},
    10: {'source': 10, 'destination': 11, 'fiber_length': 59.3806692091606},
    11: {'source': 12, 'destination': 13, 'fiber_length': 247.269660681376},
    12: {'source': 13, 'destination': 31, 'fiber_length': 13.3770556005499},
    13: {'source': 31, 'destination': 32, 'fiber_length': 16.9657329302718},
    14: {'source': 32, 'destination': 13, 'fiber_length': 7.80966628245527},
    15: {'source': 13, 'destination': 14, 'fiber_length': 70.3315947809901},
    16: {'source': 14, 'destination': 15, 'fiber_length': 48.123221602719},
    17: {'source': 15, 'destination': 16, 'fiber_length': 39.0426284677274},
    18: {'source': 16, 'destination': 17, 'fiber_length': 8.21726053561399},
    19: {'source': 17, 'destination': 18, 'fiber_length': 11.1144345141296},
    20: {'source': 18, 'destination': 19, 'fiber_length': 70.276533236148},
    21: {'source': 19, 'destination': 20, 'fiber_length': 277.996528654729},
    22: {'source': 20, 'destination': 21, 'fiber_length': 215.532736950386},
    23: {'source': 21, 'destination': 23, 'fiber_length': 54.4717060313279},
    24: {'source': 21, 'destination': 22, 'fiber_length': 39.5426001229303},
    25: {'source': 21, 'destination': 24, 'fiber_length': 69.0834229176525},
    26: {'source': 21, 'destination': 26, 'fiber_length': 33.166962140587},
    27: {'source': 21, 'destination': 27, 'fiber_length': 51.6479216007335},
    28: {'source': 22, 'destination': 13, 'fiber_length': 75.5737447884841},
    29: {'source': 22, 'destination': 13, 'fiber_length': 75.5737447884841},
    30: {'source': 24, 'destination': 25, 'fiber_length': 45.0690148456258},
    31: {'source': 25, 'destination': 26, 'fiber_length': 80.9597497911383},
    32: {'source': 26, 'destination': 21, 'fiber_length': 33.166962140587},
    33: {'source': 21, 'destination': 27, 'fiber_length': 51.6479216007335},
    34: {'source': 27, 'destination': 28, 'fiber_length': 176.928217229805},
    35: {'source': 28, 'destination': 29, 'fiber_length': 53.6811414191371},
    36: {'source': 29, 'destination': 30, 'fiber_length': 65.1837079627735},
    37: {'source': 30, 'destination': 1, 'fiber_length': 67.3003276131565},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
