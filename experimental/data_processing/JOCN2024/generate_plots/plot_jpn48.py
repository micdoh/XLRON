import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from io import StringIO
from networkx.drawing.layout import spring_layout

cost239_nodes = """
Node,Latitude,Longitude,Location Name,Country
1,51.50,-0.12,London,UK
2,48.85,2.35,Paris,France
3,50.85,4.35,Brussels,Belgium
4,52.37,4.89,Amsterdam,Netherlands
5,49.61,6.13,Luxembourg,Luxembourg
6,47.37,8.54,Zurich,Switzerland
7,45.46,9.18,Milan,Italy
8,55.67,12.56,Copenhagen,Denmark
9,52.52,13.41,Berlin,Germany
10,50.08,14.43,Prague,Czech Republic
11,48.21,16.37,Vienna,Austria
"""

cost239_edges = """Edge_ID,Source,Destination,Computed Length (km)
1,1,2,900
2,1,3,780
3,1,4,1100
4,1,8,2620
5,2,3,600
6,2,5,800
7,2,6,1200
8,2,7,1640
9,2,9,2180
10,3,4,420
11,3,5,440
12,3,7,1860
13,4,5,780
14,4,8,1520
15,4,9,1320
16,5,6,700
17,5,10,1460
18,6,7,640
19,6,10,1130
20,6,11,1460
21,7,11,1640
22,8,9,780
23,8,10,1480
24,9,10,680
25,9,11,1320
26,10,11,640"""

nsfnet_nodes = """
Node,Latitude,Longitude,Location Name,Country
1,47.61,-122.33,Seattle WA,USA
2,37.77,-122.41,San Francisco CA1,USA
3,34.05,-118.24,Los Angeles CA2,USA
4,40.76,-111.89,Salt Lake City UT,USA
5,39.74,-104.99,Denver CO,USA
6,29.76,-95.36,Houston TX,USA
7,41.25,-95.93,Omaha NE,USA
8,41.88,-87.62,Chicago IL,USA
9,40.44,-79.99,Pittsburgh PA,USA
10,33.74,-84.39,Atlanta GA,USA
11,42.33,-83.04,Detroit MI,USA
12,42.00,-74.00,New York NY,USA
13,40.72,-74.17,Newark NJ,USA
14,38.90,-77.03,Washington DC,USA
"""

nsfnet_edges = """
Edge_ID,Source,Destination,Computed Length (km)
1,1,2,1050
2,1,3,1500
3,1,8,2400
4,2,3,600
5,2,4,750
6,3,6,1800
7,4,5,600
8,4,11,1950
9,5,6,1200
10,5,7,600
11,6,10,1050
12,6,14,1800
13,7,8,750
14,7,10,1350
15,8,9,750
16,9,10,750
17,9,12,300
18,9,13,300
19,11,12,600
20,11,13,750
21,12,14,300
22,13,14,150
"""

usnet_nodes = """
Node,Latitude,Longitude,Location Name,Country
1,47.6062,-122.3321,Seattle,USA
2,45.5155,-122.6789,Portland,USA
3,37.7749,-122.4194,San Francisco,USA
4,36.1699,-115.1398,Las Vegas,USA
5,34.0522,-118.2437,Los Angeles,USA
6,46.8772,-96.7898,Fargo,USA
7,39.7392,-104.9903,Denver,USA
8,33.4484,-112.0740,Phoenix,USA
9,41.2565,-96.7898,Omaha,USA
10,35.0844,-106.6504,Albuquerque,USA
11,43.0389,-87.9065,Milwaukee,USA
12,38.6270,-90.1994,St Louis,USA
13,32.7767,-96.7970,Dallas,USA
14,29.7604,-95.3698,Houston,USA
15,40.4406,-79.9959,Pittsburgh,USA
16,36.1627,-86.7816,Nashville,USA
17,32.3792,-86.3077,Montgomery,USA
18,30.6954,-88.0399,Mobile,USA
19,42.6526,-73.7562,Albany,USA
20,42.3601,-71.0589,Boston,USA
21,38.9072,-77.0369,Washington DC,USA
22,35.2271,-80.8431,Charlotte,USA
23,33.7490,-84.3880,Atlanta,USA
24,25.7617,-80.1918,Miami,USA
"""

usnet_edges_short = """
Edge_ID,Source,Destination,Computed Length (km)
1,1,2,252
2,1,6,364
3,2,3,216
4,2,6,360
5,3,4,276
6,3,5,432
7,3,7,304
8,4,5,220
9,4,7,280
10,5,8,368
11,6,7,292
12,6,9,360
13,6,11,572
14,7,8,464
15,7,9,328
16,8,10,272
17,9,10,440
18,9,11,364
19,9,12,320
20,10,13,320
21,10,14,268
22,11,12,288
23,11,15,344
24,11,19,648
25,12,13,236
26,12,16,280
27,13,14,244
28,13,17,332
29,14,18,388
30,15,16,228
31,15,20,272
32,16,17,224
33,16,21,288
34,16,22,272
35,17,18,280
36,17,22,168
37,17,23,364
38,18,24,280
39,19,20,188
40,20,21,216
41,21,22,164
42,22,23,260
43,23,24,180
"""

usnet_edges = """
Edge_ID,Source,Destination,Computed Length (km)
1,1,2,800
2,1,6,1000
3,2,3,1100
4,2,6,950
5,3,4,250
6,3,5,1000
7,3,7,1000
8,4,5,800
9,4,7,850
10,5,8,1200
11,6,7,1000
12,6,9,1200
13,6,11,1900
14,7,8,1150
15,7,9,1000
16,8,10,900
17,9,10,1000
18,9,11,1400
19,9,12,1000
20,10,13,950
21,10,14,850
22,11,12,900
23,11,15,1300
24,11,19,2800
25,12,13,1000
26,12,16,1100
27,13,14,650
28,13,17,1100
29,14,18,1200
30,15,16,800
31,15,20,1300
32,16,17,1000
33,16,21,1000
34,16,22,800
35,17,18,800
36,17,22,850
37,17,23,1000
38,18,24,900
39,19,20,700
40,20,21,700
41,21,22,300
42,22,23,600
43,23,24,900
"""

jpn48_nodes = """
Node,Latitude,Longitude,Location Name,Country
1,43.075,141.34,Sapporo,Japan
2,40.82,140.73,Aomori,Japan
3,39.7,141.149,Morioka,Japan
4,38.27,140.85,Sendai,Japan
5,39.72,140.09,Akita,Japan
6,38.26,140.34,Yamagata,Japan
7,37.77,140.46,Fukushima,Japan
8,36.37,140.47,Mito,Japan
9,36.57,139.88,Utsunomiya,Japan
10,36.4,139.06,Maebashi,Japan
11,35.9,139.63,Omiya,Japan
12,35.6,140.16,Chiba,Japan
13,35.68,139.77,Tokyo,Japan
14,35.66,139.28,Hachioji,Japan
15,35.458,139.64,Yokohama,Japan
16,37.92,139.05,Niigata,Japan
17,36.71,137.209,Toyama,Japan
18,36.58,136.65,Kanazawa,Japan
19,36.07,136.22,Fukui,Japan
20,35.67,138.56,Kofu,Japan
21,36.66,138.19,Nagano,Japan
22,35.45,136.76,Gifu,Japan
23,35.013,138.4,Shizuoka,Japan
24,35.149,136.91,Nagoya,Japan
25,34.73,136.509,Tsu,Japan
26,35.03,135.85,Otsu,Japan
27,35.018,135.76,Kyoto,Japan
28,34.67,135.5,Osaka,Japan
29,34.69,135.18,Kobe,Japan
30,34.67,135.89,Nara,Japan
31,34.25,135.18,Wakayama,Japan
32,35.51,134.22,Tottori,Japan
33,35.48,133.05,Matsue,Japan
34,34.66,133.92,Okayama,Japan
35,34.39,132.449,Hiroshima,Japan
36,34.23,131.45,Yamaguchi,Japan
37,34.05,134.52,Tokushima,Japan
38,34.34,134.05,Takamatsu,Japan
39,33.84,132.75,Matsuyama,Japan
40,33.58,133.53,Kochi,Japan
41,33.57,130.35,Hakata,Japan
42,33.26,130.055,Saga,Japan
43,32.75,129.88,Nagasaki,Japan
44,32.81,130.7,Kumamoto,Japan
45,33.25,131.6,Oita,Japan
46,31.91,131.42,Miyazaki,Japan
47,31.6,130.559,Kagoshima,Japan
48,26.21,127.68,Naha,Japan
"""

jpn48_edges = """
Edge_ID,Source,Destination,Computed Length (km)
1,1,2,714.3
2,1,3,614.7
3,2,3,267.6
4,2,5,272.85
5,3,5,190.95
6,3,4,275.25
7,5,6,317.4
8,4,6,91.65
9,5,16,409.5
10,6,16,280.5
11,4,7,118.5
12,4,8,368.1
13,7,16,270.15
14,7,9,244.95
15,9,8,143.4
16,8,12,191.25
17,8,11,175.5
18,9,11,118.8
19,9,10,159.75
20,11,12,99.15
21,10,11,112.05
22,12,13,58.8
23,11,13,45.45
24,13,15,43.2
25,13,14,71.1
26,14,15,54.75
27,10,14,144.6
28,15,23,227.1
29,14,20,130.05
30,10,21,176.1
31,20,23,183.6
32,10,16,343.35
33,16,17,381.15
34,16,21,316.95
35,20,21,246
36,17,18,89.1
37,17,21,289.2
38,21,24,376.2
39,20,24,394.2
40,23,24,278.7
41,18,19,115.05
42,19,27,222.15
43,22,24,45.45
44,22,26,160.95
45,24,25,99.75
46,25,26,126.6
47,25,31,548.1
48,25,30,134.25
49,26,27,15
50,27,30,62.55
51,28,30,78
52,27,28,58.5
53,27,32,380.25
54,27,29,116.1
55,28,29,55.35
56,28,31,114.15
57,29,34,215.1
58,37,38,111.75
59,34,38,107.7
60,31,37,98.7
61,37,40,235.05
62,38,40,238.95
63,32,34,212.7
64,38,39,291.6
65,34,35,241.95
66,32,33,182.4
67,39,40,376.8
68,33,36,384.75
69,35,36,199.2
70,35,39,99.3
71,39,45,249.75
72,36,41,221.85
73,41,45,297.75
74,41,42,80.4
75,41,44,177.6
76,44,45,222
77,45,46,310.5
78,42,43,150.45
79,44,47,255.75
80,46,47,188.85
81,43,48,1137
82,47,48,1010.55
"""

if __name__ == '__main__':
    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.ravel()  # Flatten axes array for easier indexing

    # Create subplot labels
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']

    # Define label positions for clockwise order (top-left, top-right, bottom-right, bottom-left)
    positions = [
        (0.3, 0.95, 'left', 'top'),  # top left
        (0.6, 0.95, 'right', 'top'),  # top right
        (0.3, 0.05, 'left', 'bottom'),  # bottom right
        (0.6, 0.05, 'right', 'bottom')  # bottom left
    ]

    # Process each network
    for idx, (name, nodes, edges) in enumerate([
        ('NSFNET', nsfnet_nodes, nsfnet_edges),
        ('COST239', cost239_nodes, cost239_edges),
        ('USNET', usnet_nodes, usnet_edges),
        ('JPN48', jpn48_nodes, jpn48_edges)
    ]):
        # Read CSV files
        nodes = pd.read_csv(StringIO(nodes))
        edges = pd.read_csv(StringIO(edges))

        # Create a graph
        G = nx.Graph()

        # Add nodes
        for _, row in nodes.iterrows():
            G.add_node(row['Node'], pos=(row['Longitude'], row['Latitude']))

        # Add edges
        for _, row in edges.iterrows():
            G.add_edge(row['Source'], row['Destination'],
                       weight=row['Computed Length (km)'],
                       inverse_weight=1 / row['Computed Length (km)'])

        # Get node positions
        initial_pos = nx.get_node_attributes(G, 'pos')
        if name == 'JPN48':
            fixed_nodes = [1, 2, 3, 4, 5, 16, 17, 18, 19, 21, 25, 40, 33, 46, 36, 47, 48, 43]
        else:
            fixed_nodes = G.nodes()
        pos = spring_layout(G, k=0.12, iterations=100, pos=initial_pos,
                            weight='inverse_weight', fixed=fixed_nodes)

        # Clear the current subplot
        axes[idx].clear()

        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='black', width=2, ax=axes[idx])

        # Draw nodes and labels
        node_size = 1500
        node_color = 'white'
        nodeLineWidth = 2
        font_size = 22

        for node in G.nodes():
            nx.draw_networkx_nodes(G, pos, nodelist=[node],
                                   node_size=node_size,
                                   node_color=node_color,
                                   edgecolors='black',
                                   linewidths=nodeLineWidth,
                                   alpha=1,
                                   ax=axes[idx])

            nx.draw_networkx_labels(G, pos,
                                    labels={node: str(node)},
                                    font_size=font_size,
                                    font_family='Arial',
                                    font_weight='bold',
                                    ax=axes[idx])

        # Add edge labels
        edge_labels = {(u, v): f"{round(d['weight'], -1):.0f}"
                       for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                     font_size=13,
                                     font_family='Arial',
                                     font_weight='bold',
                                     ax=axes[idx])

        # Remove axes
        axes[idx].set_axis_off()

        # Get position for current subplot
        x, y, halign, valign = positions[idx]

        # Add subplot label and network name
        axes[idx].text(x, y, f"{subplot_labels[idx]} {name}",
                       transform=axes[idx].transAxes,
                       fontsize=40,
                       fontweight='bold',
                       family='Arial',
                       horizontalalignment=halign,
                       verticalalignment=valign,
                       bbox=dict(facecolor='white',
                                 alpha=0.8,
                                 edgecolor='none',
                                 pad=5))

        # Print graph stats
        print(f"Graph: {name}")
        print(f"Nodes: {G.number_of_nodes()}")
        print(f"Edges: {G.number_of_edges()}")
        print(f"Diameter: {nx.diameter(G)}")
        print(f"Global efficiency: {nx.global_efficiency(G):.2f}")
        print(f"Local efficiency: {nx.local_efficiency(G):.2f}")
        print(f"Transitivity: {nx.transitivity(G):.2f}")
        print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
        print(f"Average clustering coefficient: {nx.average_clustering(G):.2f}")
        print(f"Average shortest path length: {nx.average_shortest_path_length(G):.2f}")
        print(f"Connectivity: {nx.node_connectivity(G):.2f}")
        print(f"Average node connectivity: {nx.average_node_connectivity(G):.2f}")
        print(f"Edge connectivity: {nx.edge_connectivity(G)}")
        print(f"Mean edge length: {sum([d['weight'] for u, v, d in G.edges(data=True)]) / G.number_of_edges():.2f}")
        print("Min edge length:", min([d['weight'] for u, v, d in G.edges(data=True)]))
        print("Max edge length:", max([d['weight'] for u, v, d in G.edges(data=True)]), "\n\n")

    # Adjust the layout
    plt.tight_layout()
    plt.savefig('networks_plots_short.png')
    plt.show()
