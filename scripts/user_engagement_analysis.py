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
        self.cluster_centers = None

    def aggregate_metrics(self):
        """Aggregate metrics per customer ID (MSISDN)."""
        self.metrics = self.data.groupby('MSISDN/Number').agg({
            'Dur. (ms)': 'sum',                    # Total duration of sessions
            'Total DL (Bytes)': 'sum',             # Total download traffic
            'Total UL (Bytes)': 'sum',             # Total upload traffic
            'MSISDN/Number': 'size'                # Session frequency (size counts rows per group)
        }).rename(columns={
            'Dur. (ms)': 'total_session_duration',
            'Total DL (Bytes)': 'total_download_traffic',
            'Total UL (Bytes)': 'total_upload_traffic',
            'MSISDN/Number': 'sessions_frequency'
        }).reset_index()

    def report_top_customers(self):
        """Report the top 10 customers per engagement metric."""
        top_10_sessions = self.metrics.nlargest(10, 'sessions_frequency')
        top_10_duration = self.metrics.nlargest(10, 'total_session_duration')
        top_10_download = self.metrics.nlargest(10, 'total_download_traffic')
        top_10_upload = self.metrics.nlargest(10, 'total_upload_traffic')
        return top_10_sessions, top_10_duration, top_10_download, top_10_upload

    def normalize_and_cluster(self, n_clusters=3):
        """Normalize the metrics and perform K-means clustering."""
        scaler = StandardScaler()
        features = ['sessions_frequency', 'total_session_duration', 
                    'total_download_traffic', 'total_upload_traffic']
        self.normalized_metrics = scaler.fit_transform(self.metrics[features])
        
        # Perform K-means clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.metrics['cluster'] = self.kmeans.fit_predict(self.normalized_metrics)
        
        # Calculate cluster centers and inverse transform
        self.cluster_centers = pd.DataFrame(scaler.inverse_transform(self.kmeans.cluster_centers_), columns=features)
        self.cluster_centers['cluster'] = range(n_clusters)

    def cluster_summary(self):
        """Compute summary statistics for each cluster."""
        cluster_summary = self.metrics.groupby('cluster').agg({
            'sessions_frequency': ['min', 'max', 'mean', 'sum'],
            'total_session_duration': ['min', 'max', 'mean', 'sum'],
            'total_download_traffic': ['min', 'max', 'mean', 'sum'],
            'total_upload_traffic': ['min', 'max', 'mean', 'sum']
        }).reset_index()

        # Flatten the multi-level column index
        cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns.values]
        
        return cluster_summary

    def aggregate_traffic_per_application(self, applications):
        """Aggregate traffic data per application."""
        app_traffic = pd.DataFrame()

        for app, columns in applications.items():
            app_dl, app_ul = columns
            temp_traffic = self.data.groupby('MSISDN/Number').agg({
                app_dl: 'sum',
                app_ul: 'sum'
            }).reset_index().rename(columns={app_dl: 'download_bytes', app_ul: 'upload_bytes', 'MSISDN/Number': 'MSISDN'})
            temp_traffic['application'] = app
            app_traffic = pd.concat([app_traffic, temp_traffic], ignore_index=True)
        
        # Compute total traffic per application
        app_total_traffic = app_traffic.groupby('application').agg({
            'download_bytes': 'sum',
            'upload_bytes': 'sum'
        }).reset_index()
        app_total_traffic['total_bytes'] = app_total_traffic['download_bytes'] + app_total_traffic['upload_bytes']
        
        # Get top 10 most engaged users per application
        app_traffic['total_bytes'] = app_traffic['download_bytes'] + app_traffic['upload_bytes']
        top_10_engaged_per_app = app_traffic.groupby('application').apply(lambda x: x.nlargest(10, 'total_bytes')).reset_index(drop=True)
        return app_total_traffic, top_10_engaged_per_app

    def plot_top_applications(self, top_3_apps):
        """Plot the top 3 most used applications."""
        plt.figure(figsize=(12, 6))
        sns.barplot(x='application', y='total_bytes', data=top_3_apps)
        plt.title('Top 3 Most Used Applications by Total Traffic')
        plt.xlabel('Application')
        plt.ylabel('Total Traffic (Bytes)')
        plt.show()

    def elbow_method(self):
        """Determine optimal k using the Elbow Method."""
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(self.normalized_metrics)
            wcss.append(kmeans.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, 11), wcss, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.show()