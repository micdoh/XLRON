def create_railtel_india_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 31.3323762, 'long': 75.576889, 'location': 'Jalandhar', 'country': 'India'},
    2: {'lat': 30.72984395, 'long': 76.78414567016054, 'location': 'Chandigarh', 'country': 'India'},
    3: {'lat': 28.6273928, 'long': 77.1716954, 'location': 'Karol Bagh', 'country': 'India'},
    4: {'lat': 26.9154576, 'long': 75.8189817, 'location': 'Jaipur', 'country': 'India'},
    5: {'lat': 23.0216238, 'long': 72.5797068, 'location': 'Ahmedabad', 'country': 'India'},
    6: {'lat': 26.8381, 'long': 80.9346001, 'location': 'Lucknow', 'country': 'India'},
    7: {'lat': 19.0785451, 'long': 72.878176, 'location': 'Mumbai', 'country': 'India'},
    8: {'lat': 25.6093239, 'long': 85.1235252, 'location': 'Patna', 'country': 'India'},
    9: {'lat': 21.1498134, 'long': 79.0820556, 'location': 'Nagpur', 'country': 'India'},
    10: {'lat': 22.5726459, 'long': 88.3638953, 'location': 'Kolkata', 'country': 'India'},
    11: {'lat': 20.2602964, 'long': 85.8394521, 'location': 'Bhubaneshwar', 'country': 'India'},
    12: {'lat': 18.521428, 'long': 73.8544541, 'location': 'Pune', 'country': 'India'},
    13: {'lat': 17.360589, 'long': 78.4740613, 'location': 'Hyderabad', 'country': 'India'},
    14: {'lat': 12.98815675, 'long': 77.62260003796, 'location': 'Bangalore', 'country': 'India'},
    15: {'lat': 12.8698101, 'long': 74.8430082, 'location': 'Mangalore', 'country': 'India'},
    16: {'lat': 12.3051828, 'long': 76.6553609, 'location': 'Mysore', 'country': 'India'},
    17: {'lat': 13.0836939, 'long': 80.270186, 'location': 'Chennai', 'country': 'India'},
    18: {'lat': 11.0018115, 'long': 76.9628425, 'location': 'Coimbatore', 'country': 'India'},
    19: {'lat': 9.9674277, 'long': 76.2454436, 'location': 'Cochin', 'country': 'India'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 199.6756030431534},
    2.0: {'source': 2.0, 'destination': 3.0, 'fiber_length': 355.1404781790791},
    3.0: {'source': 3.0, 'destination': 4.0, 'fiber_length': 348.3957409393138},
    4.0: {'source': 4.0, 'destination': 5.0, 'fiber_length': 813.3462757027237},
    5.0: {'source': 5.0, 'destination': 6.0, 'fiber_length': 1414.440283371722},
    6.0: {'source': 6.0, 'destination': 7.0, 'fiber_length': 1500.0},
    7.0: {'source': 7.0, 'destination': 8.0, 'fiber_length': 1815.705151024932},
    8.0: {'source': 8.0, 'destination': 9.0, 'fiber_length': 1186.577856908703},
    9.0: {'source': 9.0, 'destination': 10.0, 'fiber_length': 1456.008537089643},
    10.0: {'source': 10.0, 'destination': 11.0, 'fiber_length': 549.8806056586044},
    11.0: {'source': 11.0, 'destination': 12.0, 'fiber_length': 1589.428759726822},
    12.0: {'source': 12.0, 'destination': 13.0, 'fiber_length': 758.1533396885486},
    13.0: {'source': 13.0, 'destination': 14.0, 'fiber_length': 742.0494867757965},
    14.0: {'source': 14.0, 'destination': 15.0, 'fiber_length': 452.2899035940579},
    15.0: {'source': 15.0, 'destination': 16.0, 'fiber_length': 309.685737874238},
    16.0: {'source': 16.0, 'destination': 17.0, 'fiber_length': 602.340119649406},
    17.0: {'source': 17.0, 'destination': 18.0, 'fiber_length': 641.5580476680149},
    18.0: {'source': 18.0, 'destination': 19.0, 'fiber_length': 208.8273757799214},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
