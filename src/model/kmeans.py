import numpy as np


class Kmeans:
    def __init__(self, groups, histogram):
        self.groups = groups
        self.points = histogram

    # K-means clustering
    def process(self):
        centroids = np.random.randint(0, 256, size=self.groups)
        clusters = np.zeros((256,), dtype=np.int)

        centroids_old = centroids + 1
        while not np.array_equal(centroids, centroids_old):
            centroids_old = np.copy(centroids)

            clusters = self.cluster(clusters, centroids)
            centroids = self.get_centroids(clusters, centroids)

        return clusters

    # Cluster data
    def cluster(self, clusters, centroids):
        for i in range(256):
            min_dist = 0
            for k in range(self.groups):
                distance = abs(self.points[i] - centroids[k])

                if k == 0 or distance <= min_dist:
                    min_dist = distance
                    clusters[i] = k

        return clusters

    # Recalculate centroids
    def get_centroids(self, clusters, centroids):
        for k in range(self.groups):
            if sum(clusters == k) > 0:
                centroids[k] = sum(self.points[clusters == k]) / sum(clusters == k)

        return centroids
