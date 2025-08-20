# src/features_layout.py

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix, ConvexHull

def extract_coordinates(df):
    """Extract WEC (X, Y) layout coordinates from column pairs like X1, Y1, ..., XN, YN."""
    x_cols = sorted([col for col in df.columns if col.startswith('X')])
    y_cols = sorted([col for col in df.columns if col.startswith('Y')])

    assert len(x_cols) == len(y_cols), "Mismatch between X and Y columns"

    num_wecs = len(x_cols)
    coords = []

    for i in range(len(df)):
        row_coords = []
        for j in range(num_wecs):
            x_val = df.iloc[i][x_cols[j]]
            y_val = df.iloc[i][y_cols[j]]
            if not (np.isnan(x_val) or np.isnan(y_val)):
                row_coords.append([x_val, y_val])
        coords.append(np.array(row_coords))

    return coords


def mean_nearest_distance(coords):
    """Mean nearest WEC distance per layout."""
    result = []
    for layout in coords:
        dists = distance_matrix(layout, layout)
        np.fill_diagonal(dists, np.inf)
        nearest_dists = np.min(dists, axis=1)
        result.append(np.mean(nearest_dists))
    return np.array(result)


def min_inter_wec_distance(coords):
    """Minimum inter-WEC distance per layout."""
    return np.array([
        np.min(distance_matrix(layout, layout)[np.triu_indices(layout.shape[0], k=1)])
        for layout in coords
    ])


def cluster_density(coords, radius=200):
    """Average number of WECs within a radius of each WEC."""
    densities = []
    for layout in coords:
        count = 0
        for i in range(layout.shape[0]):
            dists = np.linalg.norm(layout - layout[i], axis=1)
            count += np.sum((dists < radius) & (dists > 0))  # exclude self
        densities.append(count / layout.shape[0])
    return np.array(densities)


def convex_hull_area(coords):
    """Convex hull area of each layout."""
    areas = []
    for layout in coords:
        try:
            hull = ConvexHull(layout)
            areas.append(hull.volume)
        except:
            areas.append(0.0)
    return np.array(areas)


def layout_symmetry_score(coords):
    """Ratio of std X vs std Y as a proxy for symmetry."""
    return np.array([
        np.std(layout[:, 0]) / (np.std(layout[:, 1]) + 1e-6)  # prevent div by zero
        for layout in coords
    ])


def add_layout_features(df):
    """Main function to compute and append all spatial features to dataframe."""
    coords = extract_coordinates(df)

    df = df.copy()
    df['MeanNearestWEC_Dist'] = mean_nearest_distance(coords)
    df['MinWEC_Dist'] = min_inter_wec_distance(coords)
    df['WEC_Cluster_Density'] = cluster_density(coords)
    df['ConvexHull_Area'] = convex_hull_area(coords)
    df['Layout_Symmetry_Score'] = layout_symmetry_score(coords)

    return df
