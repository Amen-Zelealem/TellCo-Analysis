# test_experience_analytics.py

import pandas as pd
import numpy as np
import unittest
from scripts.experience_analysis import ExperienceAnalytics

class TestExperienceAnalytics(unittest.TestCase):
    def setUp(self):
        # Create a mock dataset for testing
        data = {
            'MSISDN/Number': [1, 2, 3, 4, 5],
            'TCP DL Retrans. Vol (Bytes)': [100, np.nan, 150, 200, 250],
            'TCP UL Retrans. Vol (Bytes)': [50, 60, np.nan, 70, 80],
            'Avg RTT DL (ms)': [100, 110, np.nan, 130, 140],
            'Avg RTT UL (ms)': [60, 70, 65, np.nan, 85],
            'Avg Bearer TP DL (kbps)': [2000, 2100, 2200, np.nan, 2300],
            'Avg Bearer TP UL (kbps)': [1200, 1300, 1400, 1500, np.nan],
            'Handset Type': ['iPhone', 'Samsung', np.nan, 'Huawei', 'iPhone']
        }
        self.df = pd.DataFrame(data)
        self.experience_analytics = ExperienceAnalytics(self.df)

    def test_aggregate_user_experience(self):
        # Call the method being tested
        user_agg = self.experience_analytics.aggregate_user_experience()

        # Print aggregated results for debugging
        print("Aggregated DataFrame:")
        print(user_agg)

        # Print expected results for debugging
        expected_tcp_retransmission = [150.0, 235.0, 215.0, 270.0, 330.0]
        expected_rtt = [80.0, 90.0, 92.5, 100.0, 112.5]
        expected_throughput = [1600.0, 1700.0, 1800.0, 1825.0, 1825.0]


        # Print expected results for debugging
        print("Expected TCP Retransmission:")
        print(expected_tcp_retransmission)
        print("Expected RTT:")
        print(expected_rtt)
        print("Expected Throughput:")
        print(expected_throughput)

        # Test if the aggregated data matches the expected values
        self.assertListEqual(user_agg['TCP Retransmission'].tolist(), expected_tcp_retransmission)
        self.assertListEqual(user_agg['RTT'].tolist(), expected_rtt)
        self.assertListEqual(user_agg['Throughput'].tolist(), expected_throughput)

        # Test 4: Ensure intermediate columns are dropped
        self.assertNotIn('TCP DL Retrans. Vol (Bytes)', user_agg.columns)
        self.assertNotIn('TCP UL Retrans. Vol (Bytes)', user_agg.columns)
        self.assertNotIn('Avg RTT DL (ms)', user_agg.columns)
        self.assertNotIn('Avg RTT UL (ms)', user_agg.columns)
        self.assertNotIn('Avg Bearer TP DL (kbps)', user_agg.columns)
        self.assertNotIn('Avg Bearer TP UL (kbps)', user_agg.columns)


    def test_get_top_bottom_most_frequent(self):
        # Call the method to get top, bottom, and most frequent TCP retransmission values
        top_tcp, bottom_tcp, most_freq_tcp = self.experience_analytics.get_top_bottom_most_frequent('TCP Retransmission')

        # Filter out NaN values from top_tcp and bottom_tcp
        top_tcp = top_tcp.dropna()
        bottom_tcp = bottom_tcp.dropna()

        # Convert the results to lists for easier comparison
        top_tcp_list = top_tcp.tolist()  # Convert pandas Series to list
        bottom_tcp_list = bottom_tcp.tolist()  # Convert pandas Series to list

        # Verify that the top value is as expected (should be 330.0 in this case)
        self.assertIn(330.0, top_tcp_list)  # Top value(s) should include 330.0

        # Verify that the bottom value is as expected (should be 150.0 in this case)
        self.assertIn(150.0, bottom_tcp_list)  # Bottom value(s) should include 150.0

        # Verify that the most frequent value is 150.0 (appears twice in your mock data)
        self.assertEqual(most_freq_tcp, 150.0)
        
    def test_avg_throughput_per_handset(self):
        avg_throughput = self.experience_analytics.avg_throughput_per_handset()

        # Expected averages for each handset type
        expected_avg_throughput = {
            'iPhone': 1950.0,
            'Samsung': 1700.0,
            'Huawei': 1500.0
        }

        # Convert the DataFrame to a dictionary for easier comparison
        throughput_dict = dict(zip(avg_throughput['Handset Type'], avg_throughput['Avg_Throughput']))

        # Assert that each handset type's throughput is correctly calculated
        for handset, expected_throughput in expected_avg_throughput.items():
            self.assertAlmostEqual(throughput_dict.get(handset), expected_throughput, places=2)


    def test_k_means_clustering(self):
        # Specify the features to use for clustering
        features = ['TCP Retransmission', 'RTT', 'Throughput']

        # Call the k-means clustering method
        clustered_df, cluster_centers = self.experience_analytics.k_means_clustering(features, k=3)

        # Test if the 'Cluster' column was added to the dataframe
        self.assertIn('Cluster', clustered_df.columns)

        # Test if there are exactly 3 unique clusters
        self.assertEqual(len(clustered_df['Cluster'].unique()), 3)

        # Check if the cluster centers have the correct number of features and clusters
        self.assertEqual(cluster_centers.shape, (3, len(features) + 1))  # 3 clusters + 'Cluster' column


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)