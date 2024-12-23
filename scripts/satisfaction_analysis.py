import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
import psycopg2
import pickle
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.db_conn import conn


class UserSatisfactionAnalytics:
    def __init__(self):
        self.model = None

    def compute_score(
        self, df, cluster_centers_df, features, score_column_name, target_cluster=None
    ):
        """
        Compute score based on user metrics and a specific cluster center.

        :param df: User metrics DataFrame.
        :param cluster_centers_df: Cluster centers DataFrame.
        :param features: Features for score computation.
        :param score_column_name: Name for the score column.
        :param target_cluster: Cluster number for distance calculation.
        :return: DataFrame with user ID and computed score.
        """
        if target_cluster is None:
            raise ValueError("Target cluster must be specified.")

        target_cluster_center = cluster_centers_df[
            cluster_centers_df["cluster"] == target_cluster
        ][features].values
        scores = [
            pairwise_distances([row[features].values], target_cluster_center)[0][0]
            for _, row in df.iterrows()
        ]

        df[score_column_name] = scores
        return df[["MSISDN/Number", score_column_name]]

