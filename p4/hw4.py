import geopandas
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import csv
from typing import List
from matplotlib.figure import Figure

def world_map(Z, names, K_clusters):

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

    world['name'] = world['name'].str.strip()
    names = [name.strip() for name in names]

    world['cluster'] = np.nan

    n = len(names)
    clusters = {j: [j] for j in range(n)}

    for step in range(n-K_clusters):
        cluster1 = Z[step][0]
        cluster2 = Z[step][1]

        # Create new cluster id as n + step
        new_cluster_id = n + step

        # Merge clusters
        clusters[new_cluster_id] = clusters.pop(cluster1) + clusters.pop(cluster2)

    # Assign cluster labels to countries in the world dataset
    for i, value in enumerate(clusters.values()):
        for val in value:
            world.loc[world['name'] == names[val], 'cluster'] = i

    # Plot the map
    world.plot(column='cluster', legend=True, figsize=(15, 10), missing_kwds={
        "color": "lightgrey",  # Set the color of countries without clusters
        "label": "Other countries"
    })

    # Show the plot
    plt.show()

def load_data(filepath: str) -> List[dict]:
    """load data from file in List[dict]

    Args:
        filepath (str):
            file path

    Returns:
        list:
            element is dict not OrderedDict
    """
    data = []
    with open(filepath, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(dict(row))
    return data

def calc_features(row: dict) -> np.ndarray:
    """translate elements in each row to array

    Args:
        row (dict): 
            elment from load_data()

    Returns:
        np.ndarray: 
            row -> array(row), dtype float64
    """
    keys = ["Population", "Net migration", "GDP ($ per capita)", "Literacy (%)", "Phones (per 1000)", "Infant mortality (per 1000 births)"]
    features = []
    for key in keys:
        features.append(row[key])
    features = np.array(features).astype("float64")
    return features

def hac(features: List[np.ndarray]) -> np.ndarray:
    """implement hierarchical agglomerative clustering

    Args:
        features (List[np.ndarray]): 
            np.ndarray shape -> (6,), each array is a feature from calc_features(), 
            the length of list n does not need to be the whole csv lines,

    Returns:
        np.ndarray: 
            shape(n-1, 4)
    """
    n = len(features)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i][j] = (np.linalg.norm(features[i]-features[j]))
    Z = np.zeros((n-1, 4))
    clusters_dict = {i: [i] for i in range(n)}
    max_num = np.max(distance_matrix)
    for step in range(n - 1):
        i, j = np.unravel_index(np.argmin(distance_matrix + np.eye(distance_matrix.shape[0]) * (max_num + 1)), distance_matrix.shape)
        if i > j:
            i, j = j, i
        Z[step][0] = i
        Z[step][1] = j
        Z[step][2] = distance_matrix[i][j]
        Z[step][3] = len(clusters_dict[i]) + len(clusters_dict[j])
        clusters_dict[n + step] = clusters_dict[i] + clusters_dict[j]
        recal_distance = []
        for recal_item in range(distance_matrix.shape[0]):
            recal_distance.append(min(distance_matrix[i, recal_item], distance_matrix[j, recal_item]))
        recal_distance = np.array(recal_distance)
        distance_matrix = np.vstack([distance_matrix, recal_distance])
        recal_distance = np.append(recal_distance, 0)
        distance_matrix = np.hstack([distance_matrix, recal_distance[:, np.newaxis]])
        # distance_matrix = np.delete(distance_matrix, [i,j], axis=0)
        # distance_matrix = np.delete(distance_matrix, [i,j], axis=1)
        distance_matrix[i, :] = np.inf
        distance_matrix[:, i] = np.inf
        distance_matrix[j, :] = np.inf
        distance_matrix[:, j] = np.inf
    return Z

def normalize_features(features: List[np.ndarray]) -> List[np.ndarray]:
    """normalize

    Args:
        features (List[np.ndarray]): 
            array shape (6,)

    Returns:
        List[np.ndarray]: 
            array shape (6,), dtype float64
    """
    features_array = np.array(features)
    col_min = np.min(features_array, axis=0)
    col_max = np.max(features_array, axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1  # avoid division by zero
    normalized_features = ((features_array - col_min) / col_range)
    return [np.array(row) for row in normalized_features.tolist()]

def fig_hac(Z: np.ndarray, names: np.ndarray) -> Figure:
    """matplotlib plot figure

    Args:
        Z (_type_): 
            output of hac(), shape: (n-1, 4)
        names (_type_): 
            country names
    """
    fig = plt.figure()
    dendrogram(
        Z,
        labels=names,
        leaf_rotation=90
    )
    plt.tight_layout()
    plt.title(f"N={len(names)}")
    # plt.show()
    return fig


if __name__ == "__main__":
    data = load_data("./countries.csv")
    country_names = [row["Country"] for row in data]
    features = [calc_features(row) for row in data]
    features_normalized = normalize_features(features)
    n = len(country_names)
    Z_raw = hac(features[:n])
    Z_normalized = hac(features_normalized[:n])
    fig_raw = fig_hac(Z_raw, country_names[:n])
    fig_normalized = fig_hac(Z_normalized, country_names[:n])
    
# if __name__ == "__main__":
#     import random
#     data = load_data("./countries.csv")
#     names = [row["Country"] for row in data]
#     features = [calc_features(row) for row in data]
#     features_normalized = normalize_features(features)
#     random_indices = random.sample(range(0, len(names)), 80)
#     random_names = [names[i] for i in random_indices]
#     random_features = [features_normalized[i] for i in random_indices]
#     Z = hac(random_features)
#     world_map(Z, random_names, 5)
