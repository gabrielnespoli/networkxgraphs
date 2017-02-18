import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.cluster import KMeans
import random
import pandas as pd
import networkx as nx

# http://openflights.org/data.html


def to_float(x):
    if x.isdigit():
        return float(x)
    else:
        return x


def load_airports():
    columns=['Airport ID','Name','City','Country','IATA','ICAO','Latitude','Longitude','Altitude','Timezone',
             'DST','Tz database','time zone','Type','Source']
    return pd.read_csv("airports.dat", names=columns)


def load_routes():
    columns = ["Airline", "Airline_ID", "Source_airport", "Source_airport_ID", "Destination_airport",
               "Destination_airport_ID", "Codeshare", "Stops", "Equipment"]
    return pd.read_csv("routes.dat", names=columns)


def plot_airports_location(airports):
    plt.plot(airports[:, 7], airports[:, 6], 'bo', linewidth='0.5')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.title("Longitude x Latitude")
    plt.show()


def plot_nAirports_histogram(airports):
    countries, nAirports = np.unique(airports[:, 3], return_counts=True)
    plt.hist(nAirports, 100)
    plt.show()


def cluster_per_location(airports, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=1)
    x = [float(i) for i in airports[:, 6].tolist()]
    y = [float(i) for i in airports[:, 7].tolist()]  #
    z = list(zip(x, y))
    kmeans.fit(z)
    c = kmeans.predict(z)
    return list(zip(c, z))


def plot_clusters(airports):
    n_clusters = 5
    w = cluster_per_location(airports, n_clusters)

    # define randomly the colors of the plot
    random_colors = random.sample(list(colors.cnames.items()), n_clusters)
    items_colors = [random_colors[airport_cluster[0]][1] for airport_cluster in w]

    plt.scatter(airports[:, 7], airports[:, 6], c=items_colors)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.title("Longitude x Latitude")
    plt.show(block=True)


def is_connected(routes, source_airport, destination_airport):
    print(len(routes[(routes.Source_airport==source_airport)&(routes.Destination_airport==destination_airport)]) > 0)


def create_route_graph(airports, routes):
    G = nx.Graph()
    for _, row in airports.iterrows():
        node = int(float(row['Airport ID']))
        lat = float(row['Latitude'])
        long = float(row['Longitude'])
        G.add_node(node, id=node, name=row['Name'], country=row['Country'],
                   lat=lat, long=long, region=row[11], size=10, color='blue', pos=(long, lat))

    for _, row in routes.iterrows():
        # remove garbage (some lines have garbage)
        if row['Source_airport_ID'] == "\\N" or row['Destination_airport_ID'] == "\\N":
            continue

        # get the id for the starting + ending airports
        source_id = int(row['Source_airport_ID'])
        destination_id = int(row['Destination_airport_ID'])
        # Create an edge
        G.add_edge(source_id, destination_id, airline=row['Airline'], airlineID=row['Airline_ID'])

    for node in G.nodes():
        if G.degree(node) > 50:
            G.node[node]['color'] = 'yellow'
            G.node[node]['size'] = 50
        if G.degree(node) > 100:
            G.node[node]['color'] = 'red'
            G.node[node]['size'] = 100
    return G


def draw_world(G):
    plt.clf()
    nx.draw_networkx_nodes(G, nx.get_node_attributes(G, 'pos'),
                           node_size=[v for v in nx.get_node_attributes(G, 'size').values()],
                           node_color=[v for v in nx.get_node_attributes(G, 'color').values()])
    nx.draw_networkx_edges(G, nx.get_node_attributes(G, 'pos'), width=0.2, alpha=0.5)
    plt.show()


def draw_italy(G):
    I = G.copy()
    for node in nx.nodes(I):
        if I.node[node]['country'] != "Italy":
            I.remove_node(node)

    for node in I.nodes():
        if I.node[node]['name'] == 'Fiumicino':
            I.node[node]['color'] = 'green'
            break

    plt.clf()
    nx.draw_networkx_nodes(I, nx.get_node_attributes(I, 'pos'),
                           node_size=[v for v in nx.get_node_attributes(I, 'size').values()],
                           node_color=[v for v in nx.get_node_attributes(I, 'color').values()])
    nx.draw_networkx_edges(I, nx.get_node_attributes(I, 'pos'), width=0.2, alpha=0.5)
    plt.show()


def draw_america(G):
    for node in G.nodes():
        if "AMERICA" not in (G.node[node]['region']).upper():
            G.remove_node(node)

    plt.clf()
    nx.draw_networkx_nodes(G, nx.get_node_attributes(G, 'pos'),
                           node_size=[s for s in nx.get_node_attributes(G, 'size').values()],
                           node_color=[v for v in nx.get_node_attributes(G, 'color').values()])
    nx.draw_networkx_edges(G, nx.get_node_attributes(G, 'pos'), width=0.2, alpha=0.5)
    plt.show(block=True)


def playing(airports, routes):
    G = create_route_graph(airports, routes)
    print(G[1555][344])
    G.edge[1555][344]['color'] = 'blue'  # defining a new attribute for an edge
    print(G[1555][344])
    print(G[1555])  # all connections
    print(nx.info(G))
    print(G.degree(1555))
    print(G.neighbors(1555))
    print(nx.info(G, 1555))

    print(nx.has_path(G, source=1555, target=18))  # Reykjavik
    print(nx.has_path(G, source=1555, target=3416))
    print(nx.has_path(G, source=1555, target=3878))
    print(nx.dijkstra_path(G, 1555, 18))

    print(nx.is_connected(G))
    low = min(nx.degree(G))
    high = max(nx.degree(G))
    print(low, high)

    # dh = nx.degree_histogram(G)
    # for i in range(low, len(dh)):
    #     bar = ''.join(dh[i] * ['*'])
    #     print("%2s (%2s) %s" % (i, dh[i], bar))

    I = G.copy()
    for node in nx.nodes(I):
        if (I.node[node]['country'] != "Italy"):
            I.remove_node(node)
    print(nx.info(I))

    # [len(c) for c in sorted(nx.connected_components(I), key=len, reverse=True)]
    # for c in sorted(nx.connected_components(I)):
    #     print(c)
    print(len(nx.dominating_set(I, 1555)))

    # plt.hist(sorted(nx.degree(G).values()), bins=50)
    # plt.show(block=True)
    #
    # plt.hist(sorted(nx.degree(I).values()), bins=20)
    # plt.show(block=True)
    #
    # plt.clf()
    # nx.draw(I)
    # plt.show(block=True)

    for node in I.nodes():
        I.node[node]['pos'] = (I.node[node]['long'], I.node[node]['lat'])
    # plt.clf()
    # nx.draw(I, nx.get_node_attributes(I, 'pos'))
    # plt.show()

    plt.clf()
    nx.draw_networkx_nodes(I, nx.get_node_attributes(I, 'pos'),
                           node_shape='.', node_size=50)
    nx.draw_networkx_edges(I, nx.get_node_attributes(I, 'pos'),
                           width=0.2, alpha=0.5)
    plt.show()

    for node in I.nodes():
        if (I.degree(node) > 1):
            I.node[node]['size'] = I.degree(node) * 10
        else:
            I.node[node]['size'] = 5
    plt.clf()
    nx.draw_networkx_nodes(I, nx.get_node_attributes(I, 'pos'),
                           node_size=[v for v in nx.get_node_attributes(I, 'size').values()])
    nx.draw_networkx_edges(I, nx.get_node_attributes(I, 'pos'), width=0.2, alpha=0.5)
    plt.show()

    for node in I.nodes():
        if (I.node[node]['name'] == 'Fiumicino'):
            I.node[node]['color'] = 'blue'
        elif (I.node[node]['name'] == 'Malpensa'):
            I.node[node]['color'] = 'green'
        else:
            I.node[node]['color'] = 'red'

    plt.clf()
    nx.draw_networkx_nodes(I, nx.get_node_attributes(I, 'pos'),
                           node_size=[v for v in nx.get_node_attributes(I, 'size').values()],
                           node_color=[v for v in nx.get_node_attributes(I, 'color').values()])
    nx.draw_networkx_edges(I, nx.get_node_attributes(I, 'pos'), width=0.2, alpha=0.5)
    plt.show(block=True)


def test():
    airports = load_airports()
    routes = load_routes()
    G = create_route_graph(airports, routes)
    draw_world(G)
    draw_italy(G)
    draw_america(G)
test()