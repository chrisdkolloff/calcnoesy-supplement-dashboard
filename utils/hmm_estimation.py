import pyemma

def clustering(data, k=100):
    return pyemma.coordinates.cluster_kmeans(data, k=k)


