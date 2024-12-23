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

    def compute_satisfaction_score(self, engagement_scores, experience_scores):
        # Merge engagement and experience scores and compute average satisfaction score.
        merged_df = pd.merge(engagement_scores, experience_scores, on="MSISDN/Number")
        merged_df["Satisfaction_Score"] = (
            merged_df["Engagement_Score"] + merged_df["Experience_Score"]
        ) / 2
        return merged_df

    def top_satisfied_customer(self, engagement_scores, experience_scores, top_n=10):
        # Get top N satisfied customers based on satisfaction scores.
        satisfaction_df = self.compute_satisfaction_score(
            engagement_scores, experience_scores
        )
        return satisfaction_df.sort_values(
            by="Satisfaction_Score", ascending=False
        ).head(top_n)[["MSISDN/Number", "Satisfaction_Score"]]

    def build_regression_model(self, engagement_df, experience_df, model_type="linear"):
        # Compute satisfaction scores and prepare data for regression.
        satisfaction_df = self.compute_satisfaction_score(engagement_df, experience_df)
        X = satisfaction_df[["Engagement_Score", "Experience_Score"]]
        y = satisfaction_df["Satisfaction_Score"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Initialize and train the model.
        model_map = {"ridge": Ridge(), "lasso": Lasso(), "linear": LinearRegression()}
        self.model = model_map.get(model_type, LinearRegression())
        self.model.fit(X_train, y_train)

        # Save the model.
        with open("satisfaction_model.pkl", "wb") as file:
            pickle.dump(self.model, file)

        # Evaluate the model.
        y_pred = self.model.predict(X_test)
        print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
        print(f"R-squared: {r2_score(y_test, y_pred)}")

        return self.model

   