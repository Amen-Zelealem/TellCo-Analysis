import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sqlalchemy import create_engine
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

    def perform_clustering(self, engagement_score, experience_score, n_clusters=2):
        """
        Perform K-Means clustering on engagement and experience scores.

        :param engagement_score: Engagement scores DataFrame.
        :param experience_score: Experience scores DataFrame.
        :param n_clusters: Number of clusters.
        :return: DataFrame with user IDs and their assigned cluster.
        """
        cluster_df = self.compute_satisfaction_score(engagement_score, experience_score)
        features = cluster_df[["Engagement_Score", "Experience_Score"]]

        # Fit K-Means model.
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_df["Cluster"] = kmeans.fit_predict(features)

        return cluster_df

    def export_to_postgresql(self, clustering_data):
        try:
            # Construct the database URL from environment variables
            db_name = os.getenv("DB_NAME")
            user = os.getenv("DB_USER")
            password = os.getenv("DB_PASSWORD")
            host = os.getenv("DB_HOST")
            port = os.getenv("DB_PORT", 5432)  # Default PostgreSQL port is 5432

            # Create the SQLAlchemy engine
            engine = create_engine(
                f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
            )
            connection = engine.raw_connection()
            cursor = connection.cursor()

            # Drop the table if it exists and create a new one
            cursor.execute("DROP TABLE IF EXISTS user_satisfaction_scores")
            cursor.execute("""
            CREATE TABLE user_satisfaction_scores (
                MSISDN FLOAT,
                engagement_score FLOAT,
                experience_score FLOAT,
                satisfaction_score FLOAT
            )
        """)

            # Insert data into the table.
            insert_query = """
                INSERT INTO user_satisfaction_scores (MSISDN, engagement_score, experience_score, satisfaction_score)
                VALUES (%s, %s, %s, %s)
            """
            data_to_insert = [
                (
                    float(row["MSISDN/Number"]),
                    float(row["Engagement_Score"]),
                    float(row["Experience_Score"]),
                    float(row["Satisfaction_Score"]),
                )
                for _, row in clustering_data.iterrows()
            ]
            cursor.executemany(insert_query, data_to_insert)
            connection.commit()

        except Exception as e:
            print(f"Error: {e}")
        finally:
            if connection:
                cursor.close()
                connection.close()
