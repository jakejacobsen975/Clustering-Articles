import random
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt

k = 4

def load_data():
    df = pd.read_csv('marketing_campaign.csv', sep='\t')
    df = pd.DataFrame({
            "Year_Birth": df["Year_Birth"],
            "Income": df["Income"]
         })
    df = (df-df.mean())/df.std()
    count = df.shape[0]

    # initial plot to see what it looks like
    #df.plot.scatter(x="NumDealsPurchases", y="Income")
    #df.plot.scatter(x="Year_Birth", y="Income")
    #plt.xlim(1940, 2000)
    #plt.ylim(0, 200000)
    #plt.show()

    points = []
    for i in range(count):
        x = df["Year_Birth"][i]
        y = df["Income"][i]
        if y < 20: # remove that one outlier at the top
            points.append((x, y))
    return points

def distance(p1, p2):
    import math
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def find_closest(point, points):
    closest = points[0]
    for p in points:
        if p != point and distance(p, point) < distance(closest, point):
            closest = p
    return closest

def average_point(points):
    x = 0
    y = 0
    for point in points:
        x += point[0]
        y += point[1]
    return (x/len(points), y/len(points))

def unzip_points(points):
    x = []
    y = []
    for point in points:
        x.append(point[0])
        y.append(point[1])
    return x, y

def initialize_centroids(points):
    # random centroids
    #centroids = []
    #while len(centroids) < k:
    #    point = random.choice(points)
    #    if point not in centroids:
    #        centroids.append(point)
    #        clusters[point] = []

    # find extremes
    # TODO: hardcoded for k=4
    minx = 10000
    maxx = 0
    miny = 10000
    maxy = 0
    for point in points:
        if point[0] > maxx:
            maxx = point[0]
        if point[1] > maxy:
            maxy = point[1]
        if point[0] < minx:
            minx = point[0]
        if point[1] < miny:
            miny = point[1]

    extremes = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]
    centroids = [(maxx, maxy), (maxx, miny), (minx, maxy), (minx, miny)]

    # find closest to each extreme
    for point in points:
        for i, expoint in enumerate(extremes):
            if point not in centroids and distance(point, expoint) < distance(centroids[i], expoint):
                centroids[i] = point

    return centroids

def assign_clusters(points, centroids, clusters):
    for point in points:
        nearest = centroids[0]
        for centroid in centroids:
            if distance(point, centroid) < distance(point, nearest):
                nearest = centroid
        clusters[nearest].append(point)
        # adjust centroid
        avg_point = average_point(clusters[nearest])
        new_centroid = find_closest(avg_point, clusters[nearest])
        cluster = clusters[nearest].copy()
        del clusters[nearest]
        clusters[new_centroid] = cluster
        i = centroids.index(nearest)
        centroids[i] = new_centroid

def main():
    points = load_data()

    # 1. initialize centroids
    centroids = initialize_centroids(points)
    clusters = {}
    for centroid in centroids:
        clusters[centroid] = []

    # 2. do initial point cluster assignment
    assign_clusters(points, centroids, clusters)

    # -- visualize
    colors = ['r', 'g', 'b', 'y']
    for i, centroid in enumerate(clusters):
        plt.scatter(*unzip_points(clusters[centroid]), c=colors[i])
    plt.scatter(*unzip_points(centroids), c='black')
    plt.show()

    # fix centroids and adjust clusters
    new_clusters = {}
    for point in points:
        nearest = centroids[0]
        for centroid in centroids:
            if distance(point, centroid) < distance(point, nearest):
                nearest = centroid
        if nearest not in new_clusters:
            new_clusters[nearest] = []
        new_clusters[nearest].append(point)

    # -- visualize
    colors = ['r', 'g', 'b', 'y']
    for i, centroid in enumerate(new_clusters):
        plt.scatter(*unzip_points(new_clusters[centroid]), c=colors[i])
    plt.scatter(*unzip_points(centroids), c='black')
    plt.show()

main()
