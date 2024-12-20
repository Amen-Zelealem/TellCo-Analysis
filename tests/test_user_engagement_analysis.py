import unittest
import pandas as pd
from scripts.user_engagement_analysis import (
    UserEngagementAnalysis,
)  # Assuming the class is in user_engagement_analysis.py


class TestUserEngagementAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a sample DataFrame
        cls.data = pd.DataFrame(
            {
                "MSISDN/Number": ["123", "123", "124", "124", "125"],
                "Dur. (ms)": [300000, 150000, 200000, 100000, 50000],
                "Total DL (Bytes)": [500000, 300000, 200000, 150000, 50000],
                "Total UL (Bytes)": [200000, 100000, 150000, 50000, 20000],
                "Youtube DL (Bytes)": [200000, 100000, 50000, 30000, 10000],
                "Youtube UL (Bytes)": [100000, 50000, 20000, 10000, 5000],
                "Netflix DL (Bytes)": [150000, 100000, 70000, 40000, 20000],
                "Netflix UL (Bytes)": [50000, 30000, 20000, 10000, 5000],
                "Gaming DL (Bytes)": [30000, 20000, 15000, 10000, 5000],
                "Gaming UL (Bytes)": [10000, 5000, 2000, 1000, 500],
                "Other DL (Bytes)": [10000, 5000, 3000, 2000, 1000],
                "Other UL (Bytes)": [5000, 2000, 1000, 500, 200],
            }
        )

        cls.analysis = UserEngagementAnalysis(cls.data)

    def test_aggregate_metrics(self):
        self.analysis.aggregate_metrics()
        metrics = self.analysis.metrics
        self.assertEqual(metrics.shape[0], 3)  # Check number of unique customers
        self.assertTrue("sessions_frequency" in metrics.columns)
        self.assertTrue("total_session_duration" in metrics.columns)
        self.assertTrue("total_download_traffic" in metrics.columns)
        self.assertTrue("total_upload_traffic" in metrics.columns)

    def test_report_top_customers(self):
        self.analysis.aggregate_metrics()
        top_customers = self.analysis.report_top_customers()
        self.assertEqual(top_customers[0].shape[0], 3)  # Top 10 by sessions frequency
        self.assertEqual(top_customers[1].shape[0], 3)  # Top 10 by session duration
        self.assertEqual(top_customers[2].shape[0], 3)  # Top 10 by download traffic
        self.assertEqual(top_customers[3].shape[0], 3)  # Top 10 by upload traffic

    def test_normalize_and_cluster(self):
        self.analysis.aggregate_metrics()
        self.analysis.normalize_and_cluster(n_clusters=3)
        self.assertEqual(
            self.analysis.metrics["cluster"].nunique(), 3
        )  # Check number of clusters
        self.assertTrue("cluster" in self.analysis.metrics.columns)

    def test_cluster_summary(self):
        self.analysis.aggregate_metrics()
        self.analysis.normalize_and_cluster(n_clusters=3)
        cluster_summary = self.analysis.cluster_summary()
        self.assertEqual(cluster_summary.shape[0], 3)  # Check number of clusters
        self.assertTrue("sessions_frequency" in cluster_summary.columns)
        self.assertTrue("total_session_duration" in cluster_summary.columns)
        self.assertTrue("total_download_traffic" in cluster_summary.columns)
        self.assertTrue("total_upload_traffic" in cluster_summary.columns)


if __name__ == "__main__":
    unittest.main()
