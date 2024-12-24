import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st

class UserEngagementVisualizations:
    
    def __init__(self, data, custom_colors):
        self.data = data
        self.custom_colors = custom_colors

    def plot_top_customers(self, data, metric_name):
        if isinstance(data, pd.DataFrame) and metric_name in data.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(data=data, x='MSISDN/Number', y=metric_name, palette=self.custom_colors, ax=ax)
            ax.set_xlabel('Customer')
            ax.set_ylabel(metric_name)
            ax.set_title(f'Top Customers by {metric_name}')
            plt.xticks(rotation=90)
            st.pyplot(fig)
        else:
            st.error("Invalid data format or metric name for plotting.")
            
        
    def plot_top_applications(self, top_3_apps):
        # Plot the top 3 most used applications
        plt.figure(figsize=(12, 6))
        sns.barplot(x='application', y='total_bytes', data=top_3_apps, palette=self.custom_colors)
        plt.title('Top 3 Most Used Applications by Total Traffic')
        plt.xlabel('Application')
        plt.ylabel('Total Traffic (Bytes)')
        # Display the plot
        st.pyplot(plt.gcf())
        plt.clf()  # Clear the figure for the next plot

    def plot_elbow_method(self, wcss):
        # Plot the elbow method for optimal k
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        st.pyplot(plt.gcf())
        plt.clf()  # Clear the figure for the next plot

    def plot_cluster_summary(self, cluster_summary):
        # Check if cluster_summary is a DataFrame
        if not isinstance(cluster_summary, pd.DataFrame):
            st.error("Cluster summary data is not available.")
            return

        # Ensure 'cluster' column is in the DataFrame
        if 'cluster_' not in cluster_summary.columns:
            st.error("'cluster_' column is missing from cluster summary data.")
            return

        metrics = ['sessions_frequency', 'total_session_duration', 'total_download_traffic', 'total_upload_traffic']
        
        for metric in metrics:
            # Filter columns related to the current metric
            metric_columns = [col for col in cluster_summary.columns if col.startswith(f"{metric}_")]
            
            if not metric_columns:
                st.error(f"No data available for {metric}.")
                continue
            
            # Select columns related to the current metric and reset index
            met = cluster_summary[['cluster_'] + metric_columns].set_index('cluster_').reset_index()
            met = met.melt(id_vars='cluster_', var_name='Summary Statistic', value_name='Value')
            
            # Plot using seaborn
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x='Summary Statistic', y='Value', hue='cluster_', data=met, ax=ax)
            
            ax.set_title(f'Cluster Summary for {metric.replace("_", " ").capitalize()}')
            ax.set_xlabel('Summary Statistic')
            ax.set_ylabel('Value')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)

            # Show plot in Streamlit
            st.pyplot(fig)
            plt.clf()  # Clear the figure for the next plot