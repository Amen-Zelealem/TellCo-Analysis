import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class ExperienceAnalytics:
    def __init__(self, df):
        self.df = df

    # Aggregate user experience metrics and handle missing values
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

    # Compute top, bottom, and most frequent values for a column
    def get_top_bottom_most_frequent(self, column):
        df = self.aggregate_user_experience()
        top_10 = df[column].nlargest(10)  # Largest values
        bottom_10 = df[column].nsmallest(10)  # Smallest values
        most_frequent = (
            df[column].mode().iloc[0] if not df[column].mode().empty else None
        )  # Most frequent value
        return top_10, bottom_10, most_frequent

    # Calculate average throughput per handset type
    def avg_throughput_per_handset(self):
        throughput_cols = ["Avg Bearer TP DL (kbps)", "Avg Bearer TP UL (kbps)"]
        self.df["Avg_Throughput"] = self.df[throughput_cols].mean(axis=1)
        return self.df.groupby("Handset Type")["Avg_Throughput"].mean().reset_index()

    # Calculate average TCP retransmission per handset type
    def avg_tcp_rtt_per_handset(self):
        self.df["TCP_Retransmission"] = self.df[
            ["TCP DL Retrans. Vol (Bytes)", "TCP UL Retrans. Vol (Bytes)"]
        ].sum(axis=1)
        return (
            self.df.groupby("Handset Type")["TCP_Retransmission"].mean().reset_index()
        )

    # Plot distribution of a specified metric for the top 10 handsets
    def plot_distribution(self, metric):
        if metric == "Throughput":
            avg_metric_per_handset = self.avg_throughput_per_handset()
        elif metric == "TCP Retransmission":
            avg_metric_per_handset = self.avg_tcp_rtt_per_handset()
        else:
            raise ValueError("Metric must be 'Throughput' or 'TCP Retransmission'")

        top_10_handsets = avg_metric_per_handset.sort_values(
            by=avg_metric_per_handset.columns[1], ascending=False
        ).head(10)

        sns.barplot(
            x="Handset Type", y=avg_metric_per_handset.columns[1], data=top_10_handsets
        )
        plt.title(f"Distribution of {metric} for Top 10 Handsets")
        plt.xticks(rotation=90)
        plt.show()

    # K-means clustering to segment users into experience groups
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
