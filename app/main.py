import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from sklearn.cluster import KMeans

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import modules
try:
    from scripts.experience_analysis import ExperienceAnalytics
    from scripts.handset_analysis import HandsetAnalysis
    from scripts.user_engagement_analysis import UserEngagementAnalysis
    from scripts.handset_dashboard import HandsetVisualization
    from scripts.user_engagement_dashboard import UserEngagementVisualizations
    from scripts.user_satisfaction_dashboard import SatisfactionDashboard

    st.success("Modules imported successfully! 🎉")
except ImportError as e:
    st.error(f"Error importing modules: {e}")

# Load your data
@st.cache_data
def load_data():
    data_url = os.path.join(os.path.dirname(__file__), "../data/cleaned_data.csv")
    df = pd.read_csv(data_url)
    return df

# Create a function to perform K-Means clustering and visualize the results
def perform_clustering(analytics, agg, features, n_clusters):
    clustered_df, cluster_centers_ = analytics.k_means_clustering(features, n_clusters)

    st.subheader("📊 Clustered Data")
    st.write(clustered_df)

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=clustered_df,
        x=features[0],
        y=features[1],
        hue="Cluster",
        palette="coolwarm",  # Changed color palette for better visual appeal
        ax=ax,
        style="Cluster",      # Different styles for clusters
        markers=["o", "s", "D", "X", "^", "v", "<", ">"],
    )
    ax.set_title("K-Means Clustering Results", fontsize=16, fontweight='bold')
    st.pyplot(fig)

    st.subheader("🌟 Cluster Centers")
    centers = pd.DataFrame(cluster_centers_, columns=features)
    st.write(centers)

# Streamlit app
def main():
    st.title("📈 Telecom User & Device Dashboard", anchor="top")
    custom_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    df = load_data()

    try:
        handset_analysis = HandsetAnalysis(df)
        handset_visualization = HandsetVisualization(custom_colors)
        analytics = ExperienceAnalytics(df)
    except Exception as e:
        st.error(f"Error initializing classes: {e}")

    st.sidebar.title("🧭 Navigation", anchor="sidebar")
    section = st.sidebar.radio(
        "Go to",
        ["👤 User Analysis", "💡 User Experience", "📊 Engagement Analysis", "🔍 User Satisfaction"],
    )

    if section == "👤 User Analysis":
        st.subheader("User & Handset Analysis")
        top_n = st.slider("Number of top handsets to display", 5, 20, 10)
        top_handsets = handset_analysis.top_handsets(top_n)
        handset_visualization.visualize_top_handsets(top_handsets, top_n)

        top_n_manufacturers = st.slider("Number of top manufacturers", 2, 10, 3)
        top_manufacturers = handset_analysis.top_manufacturers(top_n_manufacturers)
        handset_visualization.visualize_top_manufacturers(top_manufacturers, top_n_manufacturers)

        manufacturers = st.multiselect(
            "Select manufacturers",
            handset_analysis.top_manufacturers(top_n_manufacturers).index.tolist(),
        )
        if manufacturers:
            top_handsets_per_manufacturer = handset_analysis.top_handsets_per_manufacturer(manufacturers)
            handset_visualization.visualize_top_handsets_per_manufacturer(
                top_handsets_per_manufacturer, manufacturers, top_n_manufacturers
            )

    elif section == "💡 User Experience":
        st.subheader("User Experience Analytics")
        agg = analytics.aggregate_user_experience()
        features = st.sidebar.multiselect(
            "Select features for clustering",
            agg.columns.tolist(),
            default=["TCP Retransmission", "RTT", "Throughput"],
        )
        n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)

        if features:
            perform_clustering(analytics, agg, features, n_clusters)

    elif section == "📊 Engagement Analysis":
        st.subheader("User Engagement Analysis")
        enga_analysis = UserEngagementAnalysis(df)
        enga_analysis.aggregate_metrics()

        st.sidebar.subheader("📊 Top Customers Metrics")
        metric_choice = st.sidebar.selectbox(
            "Select Metric for Top Customers",
            ["sessions_frequency", "total_session_duration", "total_download_traffic", "total_upload_traffic"],
        )

        top_customers = enga_analysis.report_top_customers()
        engagement_vis = UserEngagementVisualizations(df, custom_colors)

        metric_map = {
            "sessions_frequency": 0,
            "total_session_duration": 1,
            "total_download_traffic": 2,
            "total_upload_traffic": 3,
        }

        if metric_choice in metric_map:
            index = metric_map[metric_choice]
            data_df = top_customers[index]
            engagement_vis.plot_top_customers(data_df, metric_choice)

        if st.sidebar.button("Show Elbow Method"):
            enga_analysis.normalize_and_cluster()
            wcss = []
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, random_state=42)
                kmeans.fit(enga_analysis.normalized_metrics)
                wcss.append(kmeans.inertia_)
            engagement_vis.plot_elbow_method(wcss)

        if st.sidebar.button("Show Cluster Summary"):
            enga_analysis.normalize_and_cluster(n_clusters=3)
            cluster_summary_df = enga_analysis.cluster_summary()
            engagement_vis.plot_cluster_summary(cluster_summary_df)

        if st.sidebar.button("Show Top 3 Apps"):
            applications = {
                "YouTube": ["Youtube DL (Bytes)", "Youtube UL (Bytes)"],
                "Netflix": ["Netflix DL (Bytes)", "Netflix UL (Bytes)"],
                "Gaming": ["Gaming DL (Bytes)", "Gaming UL (Bytes)"],
                "Other": ["Other DL (Bytes)", "Other UL (Bytes)"],
            }
            app_total_traffic, top_10_engaged_per_app = enga_analysis.aggregate_traffic_per_application(applications)
            top_3_apps = (
                top_10_engaged_per_app.groupby("application")
                .sum()
                .nlargest(3, "total_bytes")
            )
            engagement_vis.plot_top_applications(top_3_apps)

    elif section == "🔍 User Satisfaction":
        st.subheader("User Satisfaction Dashboard")
        st.write("This section will analyze satisfaction metrics like NPS, CSAT, and others.")
        satisfaction_dashboard = SatisfactionDashboard(df)

        # Update to handle ambiguous DataFrame issues:
        try:
            satisfaction_df = satisfaction_dashboard.show_satisfaction()
            # Display the DataFrame with increased size
            st.dataframe(satisfaction_df, width=1000, height=600)  # Set your desired width and height
        except ValueError as e:
            st.error(f"Error: {e}")
            st.warning("Check the dataset and ensure it does not have NaN or mismatched values in categorical fields.")
            # Handle potential data cleaning or validation here as needed.

if __name__ == "__main__":
    main()