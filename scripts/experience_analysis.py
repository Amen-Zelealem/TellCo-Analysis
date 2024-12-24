import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class UserEngagementAnalysis:
    def __init__(self, data):
        self.data = data
        self.metrics = None
        self.normalized_metrics = None
        self.kmeans = None

    def aggregate_metrics(self):
        # Aggregate metrics per customer ID (MSISDN)
        self.metrics = (
            self.data.groupby("MSISDN/Number")
            .agg(
                {
                    "Dur. (ms)": "sum",  # Total duration of sessions
                    "Total DL (Bytes)": "sum",  # Total download traffic
                    "Total UL (Bytes)": "sum",  # Total upload traffic
                }
            )
            .reset_index()
        )

        # Rename columns for clarity
        self.metrics.columns = [
            "MSISDN/Number",
            "total_session_duration",
            "total_download_traffic",
            "total_upload_traffic",
        ]
        self.metrics["sessions_frequency"] = (
            self.data.groupby("MSISDN/Number")
            .size()
            .reset_index(name="session_id")["session_id"]
        )

    def report_top_customers(self):
        # Report the top 10 customers per engagement metric
        top_10_sessions = self.metrics.nlargest(10, "sessions_frequency")
        top_10_duration = self.metrics.nlargest(10, "total_session_duration")
        top_10_download = self.metrics.nlargest(10, "total_download_traffic")
        top_10_upload = self.metrics.nlargest(10, "total_upload_traffic")
        return top_10_sessions, top_10_duration, top_10_download, top_10_upload

    def normalize_and_cluster(self, n_clusters=3):
        # Normalize the metrics
        scaler = StandardScaler()
        self.normalized_metrics = scaler.fit_transform(
            self.metrics[
                [
                    "sessions_frequency",
                    "total_session_duration",
                    "total_download_traffic",
                    "total_upload_traffic",
                ]
            ]
        )

        # Perform K-means clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.metrics["cluster"] = self.kmeans.fit_predict(self.normalized_metrics)

    def cluster_summary(self):
        # Compute min, max, average & total non-normalized metrics for each cluster
        cluster_summary = (
            self.metrics.groupby("cluster")
            .agg(
                {
                    "sessions_frequency": ["min", "max", "mean", "sum"],
                    "total_session_duration": ["min", "max", "mean", "sum"],
                    "total_download_traffic": ["min", "max", "mean", "sum"],
                    "total_upload_traffic": ["min", "max", "mean", "sum"],
                }
            )
            .reset_index()
        )
        return cluster_summary

    def elbow_method(self):
        # Determine optimal k using Elbow Method
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(self.normalized_metrics)
            wcss.append(kmeans.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, 11), wcss, marker="o")
        plt.title("Elbow Method for Optimal k")
        plt.xlabel("Number of Clusters")
        plt.ylabel("WCSS")
        plt.show()

class ExperienceAnalytics:
    def __init__(self, df):
        self.df = df

    def aggregate_user_experience(self):
        # Fill missing values
        self.df["TCP DL Retrans. Vol (Bytes)"] = self.df[
            "TCP DL Retrans. Vol (Bytes)"
        ].fillna(self.df["TCP DL Retrans. Vol (Bytes)"].mean())
        self.df["TCP UL Retrans. Vol (Bytes)"] = self.df[
            "TCP UL Retrans. Vol (Bytes)"
        ].fillna(self.df["TCP UL Retrans. Vol (Bytes)"].mean())
        self.df["Avg RTT DL (ms)"] = self.df["Avg RTT DL (ms)"].fillna(
            self.df["Avg RTT DL (ms)"].mean()
        )
        self.df["Avg RTT UL (ms)"] = self.df["Avg RTT UL (ms)"].fillna(
            self.df["Avg RTT UL (ms)"].mean()
        )
        self.df["Handset Type"] = self.df["Handset Type"].fillna(
            self.df["Handset Type"].mode()[0]
        )

        # Group by customer and compute aggregated metrics
        user_agg = (
            self.df.groupby("MSISDN/Number")
            .agg(
                {
                    "TCP DL Retrans. Vol (Bytes)": "mean",
                    "TCP UL Retrans. Vol (Bytes)": "mean",
                    "Avg RTT DL (ms)": "mean",
                    "Avg RTT UL (ms)": "mean",
                    "Avg Bearer TP DL (kbps)": "mean",
                    "Avg Bearer TP UL (kbps)": "mean",
                    "Handset Type": "first",
                }
            )
            .reset_index()
        )

        # Combine metrics for analysis
        user_agg["TCP Retransmission"] = (
            user_agg["TCP DL Retrans. Vol (Bytes)"]
            + user_agg["TCP UL Retrans. Vol (Bytes)"]
        )
        user_agg["RTT"] = (
            user_agg["Avg RTT DL (ms)"] + user_agg["Avg RTT UL (ms)"]
        ) / 2
        user_agg["Throughput"] = (
            user_agg["Avg Bearer TP DL (kbps)"] + user_agg["Avg Bearer TP UL (kbps)"]
        ) / 2

        # Drop intermediate columns
        user_agg.drop(
            columns=[
                "TCP DL Retrans. Vol (Bytes)",
                "TCP UL Retrans. Vol (Bytes)",
                "Avg RTT DL (ms)",
                "Avg RTT UL (ms)",
                "Avg Bearer TP DL (kbps)",
                "Avg Bearer TP UL (kbps)",
            ],
            inplace=True,
        )

        return user_agg

    def k_means_clustering(self, features, k=3):
        df = self.aggregate_user_experience()

        # Standardize the feature data
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[features])

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        df["Cluster"] = kmeans.fit_predict(scaled_features)

        # Cluster centers
        cluster_centers = pd.DataFrame(
            scaler.inverse_transform(kmeans.cluster_centers_), columns=features
        )
        cluster_centers["Cluster"] = range(k)

        return df, cluster_centers
