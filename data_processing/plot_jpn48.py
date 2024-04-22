import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from io import StringIO
from networkx.drawing.layout import spring_layout

nodes = """
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

edges = """
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
        G.add_edge(row['Source'], row['Destination'], weight=row['Computed Length (km)'], inverse_weight=1/row['Computed Length (km)'])

    # Get node positions
    initial_pos = nx.get_node_attributes(G, 'pos')
    fixed_nodes = [1,2,3,4,5,16,17,18,19,21,25,40,33,46, 36,47,48,43]
    pos = spring_layout(G, k=0.12, iterations=100, pos=initial_pos, weight='inverse_weight', fixed=fixed_nodes)

    # Create the plot
    plt.figure(figsize=(15, 12))

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='black', width=2)

    # Draw nodes and labels sequentially
    node_size = 1500
    node_color = 'white'
    nodeLineWidth = 2
    font_size = 22

    for node in G.nodes():
        # Draw individual node
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=node_size,
                               node_color=node_color, edgecolors='black',
                               linewidths=nodeLineWidth, alpha=1)

        # Draw individual label
        nx.draw_networkx_labels(G, pos, labels={node: str(node)},
                                font_size=font_size, font_family='Arial',
                                font_weight='bold')

    # Add edge labels (lengths rounded to nearest 10km)
    edge_labels = {(u, v): f"{round(d['weight'], -1):.0f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=13, font_family='Arial', font_weight='bold')

    # Remove axes
    plt.axis('off')

    # Show the plot
    plt.tight_layout()
    plt.show()

