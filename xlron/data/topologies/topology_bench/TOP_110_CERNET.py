def create_cernet_graph():
    # Create an empty graph
    G = nx.Graph()

    node_attributes = {
    1: {'lat': 39.9042, 'long': 116.4074, 'location': 'Beijing', 'country': 'China'},
    2: {'lat': 39.3434, 'long': 117.3616, 'location': 'Erwangzhuang', 'country': 'China'},
    3: {'lat': 38.0428, 'long': 114.5149, 'location': 'Shijiazhuang', 'country': 'China'},
    4: {'lat': 37.8706, 'long': 112.5489, 'location': 'Taiyuan', 'country': 'China'},
    5: {'lat': 40.8426, 'long': 111.7492, 'location': 'Haoxinying', 'country': 'China'},
    6: {'lat': 41.8057, 'long': 123.4315, 'location': 'Shenyang', 'country': 'China'},
    7: {'lat': 43.8163, 'long': 125.3235, 'location': 'Changchun', 'country': 'China'},
    8: {'lat': 45.8038, 'long': 126.5349, 'location': 'Harbin', 'country': 'China'},
    9: {'lat': 34.7466, 'long': 113.6254, 'location': 'Zhengzhou', 'country': 'China'},
    10: {'lat': 36.6512, 'long': 117.1201, 'location': 'Jinan', 'country': 'China'},
    11: {'lat': 36.0671, 'long': 120.3826, 'location': 'Qingdao', 'country': 'China'},
    12: {'lat': 34.3416, 'long': 108.9398, 'location': 'Zhangjiabao', 'country': 'China'},
    13: {'lat': 36.0611, 'long': 103.8343, 'location': 'Lanzhou', 'country': 'China'},
    14: {'lat': 36.6171, 'long': 101.7782, 'location': 'Xining', 'country': 'China'},
    15: {'lat': 38.4872, 'long': 106.2309, 'location': 'Yinchuan', 'country': 'China'},
    16: {'lat': 43.8256, 'long': 87.6169, 'location': 'Shuimogou', 'country': 'China'},
    17: {'lat': 31.2304, 'long': 121.4737, 'location': 'Shanghai', 'country': 'China'},
    18: {'lat': 30.2741, 'long': 120.1551, 'location': 'Hangzhou', 'country': 'China'},
    19: {'lat': 31.8612, 'long': 117.2855, 'location': 'Hefei', 'country': 'China'},
    20: {'lat': 32.0603, 'long': 118.7969, 'location': 'Meiyuan Xincun', 'country': 'China'},
    21: {'lat': 30.5928, 'long': 114.3055, 'location': "Jiang'an", 'country': 'China'},
    22: {'lat': 30.5728, 'long': 104.0668, 'location': 'Chengdu', 'country': 'China'},
    23: {'lat': 29.563, 'long': 106.5516, 'location': 'Chongqing', 'country': 'China'},
    24: {'lat': 26.647, 'long': 106.6302, 'location': 'Guiyang', 'country': 'China'},
    25: {'lat': 25.0406, 'long': 102.7123, 'location': 'Kunming', 'country': 'China'},
    26: {'lat': 28.2282, 'long': 112.9388, 'location': 'Wangyue', 'country': 'China'},
    27: {'lat': 28.682, 'long': 115.8579, 'location': 'Nanchang', 'country': 'China'},
    28: {'lat': 26.0745, 'long': 119.2965, 'location': 'Antai', 'country': 'China'},
    29: {'lat': 22.817, 'long': 108.3669, 'location': 'Nanning', 'country': 'China'},
    30: {'lat': 23.1291, 'long': 113.2644, 'location': 'Guangzhou', 'country': 'China'},
    31: {'lat': 22.5431, 'long': 114.0579, 'location': 'Shenzhen', 'country': 'China'},
    32: {'lat': 20.044, 'long': 110.1999, 'location': 'Changliu', 'country': 'China'},
    }

    edge_attributes = {
    1.0: {'source': 1.0, 'destination': 2.0, 'fiber_length': 154.1960042229768},
    2.0: {'source': 1.0, 'destination': 3.0, 'fiber_length': 395.7196398886789},
    3.0: {'source': 1.0, 'destination': 4.0, 'fiber_length': 604.8772675878472},
    4.0: {'source': 1.0, 'destination': 9.0, 'fiber_length': 935.9045379385573},
    5.0: {'source': 1.0, 'destination': 10.0, 'fiber_length': 550.535832060699},
    6.0: {'source': 1.0, 'destination': 6.0, 'fiber_length': 940.8340945122175},
    7.0: {'source': 1.0, 'destination': 12.0, 'fiber_length': 1358.118216403098},
    8.0: {'source': 2.0, 'destination': 10.0, 'fiber_length': 450.1584300849722},
    9.0: {'source': 3.0, 'destination': 9.0, 'fiber_length': 562.5929202610187},
    10.0: {'source': 3.0, 'destination': 4.0, 'fiber_length': 260.1374861086779},
    11.0: {'source': 4.0, 'destination': 9.0, 'fiber_length': 540.7627394108456},
    12.0: {'source': 4.0, 'destination': 5.0, 'fiber_length': 506.3149496829406},
    13.0: {'source': 6.0, 'destination': 7.0, 'fiber_length': 407.4716499005557},
    14.0: {'source': 7.0, 'destination': 8.0, 'fiber_length': 361.153065834294},
    15.0: {'source': 9.0, 'destination': 12.0, 'fiber_length': 647.2061489368167},
    16.0: {'source': 12.0, 'destination': 13.0, 'fiber_length': 752.4914681787538},
    17.0: {'source': 13.0, 'destination': 14.0, 'fiber_length': 291.3906846784453},
    18.0: {'source': 14.0, 'destination': 15.0, 'fiber_length': 666.1983607765019},
    19.0: {'source': 15.0, 'destination': 16.0, 'fiber_length': 2078.840198481916},
    20.0: {'source': 16.0, 'destination': 12.0, 'fiber_length': 2640.01295030407},
    21.0: {'source': 17.0, 'destination': 18.0, 'fiber_length': 247.3113150776231},
    22.0: {'source': 18.0, 'destination': 19.0, 'fiber_length': 487.9756814436001},
    23.0: {'source': 19.0, 'destination': 20.0, 'fiber_length': 216.4367408153237},
    24.0: {'source': 20.0, 'destination': 10.0, 'fiber_length': 799.7358775725775},
    25.0: {'source': 12.0, 'destination': 22.0, 'fiber_length': 930.0725305541409},
    26.0: {'source': 22.0, 'destination': 23.0, 'fiber_length': 396.2365981887913},
    27.0: {'source': 23.0, 'destination': 24.0, 'fiber_length': 486.5040101972631},
    28.0: {'source': 24.0, 'destination': 25.0, 'fiber_length': 646.2268085750065},
    29.0: {'source': 26.0, 'destination': 27.0, 'fiber_length': 434.6914991664229},
    30.0: {'source': 27.0, 'destination': 28.0, 'fiber_length': 669.6474734732757},
    31.0: {'source': 29.0, 'destination': 30.0, 'fiber_length': 753.8420698287531},
    32.0: {'source': 30.0, 'destination': 31.0, 'fiber_length': 156.3044830896164},
    33.0: {'source': 31.0, 'destination': 32.0, 'fiber_length': 730.1466179601482},
    34.0: {'source': 1.0, 'destination': 21.0, 'fiber_length': 1500.0},
    35.0: {'source': 21.0, 'destination': 30.0, 'fiber_length': 1254.46583001865},
    36.0: {'source': 21.0, 'destination': 22.0, 'fiber_length': 1469.678364115934},
    37.0: {'source': 21.0, 'destination': 20.0, 'fiber_length': 685.0597129982668},
    38.0: {'source': 21.0, 'destination': 17.0, 'fiber_length': 1031.090356422834},
    39.0: {'source': 21.0, 'destination': 9.0, 'fiber_length': 699.3653911819321},
    }

    # Add nodes
    G.add_nodes_from(node_attributes.keys())

    # Add edges
    for edge in edge_attributes.values():
        G.add_edge(edge['source'], edge['destination'], weight=edge['fiber_length'])

    # Set node attributes
    nx.set_node_attributes(G, node_attributes)

    return G
