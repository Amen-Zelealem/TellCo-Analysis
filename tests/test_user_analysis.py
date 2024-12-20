import unittest
import pandas as pd
from scripts.user_analysis import (
    UserBehaviorAnalysis,
)  # Adjust the import according to your module name


class TestUserBehaviorAnalasis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data"""
        data = {
            "Bearer Id": [1, 1, 2, 2, 3, 3],
            "Dur. (ms)": [1000, 2000, 1500, 2500, 1200, 2200],
            "Total DL (Bytes)": [500, 600, 700, 800, 900, 1000],
            "Total UL (Bytes)": [300, 400, 500, 600, 700, 800],
            "Social Media UL (Bytes)": [100, 200, 300, 400, 500, 600],
            "Social Media DL (Bytes)": [50, 60, 70, 80, 90, 100],
            "Google UL (Bytes)": [200, 300, 400, 500, 600, 700],
            "Google DL (Bytes)": [100, 120, 140, 160, 180, 200],
            "Email UL (Bytes)": [10, 20, 30, 40, 50, 60],
            "Email DL (Bytes)": [5, 6, 7, 8, 9, 10],
            "Youtube UL (Bytes)": [300, 400, 500, 600, 700, 800],
            "Youtube DL (Bytes)": [150, 160, 170, 180, 190, 200],
            "Netflix UL (Bytes)": [400, 500, 600, 700, 800, 900],
            "Netflix DL (Bytes)": [200, 220, 240, 260, 280, 300],
            "Gaming UL (Bytes)": [50, 60, 70, 80, 90, 100],
            "Gaming DL (Bytes)": [25, 30, 35, 40, 45, 50],
            "Other UL (Bytes)": [5, 10, 15, 20, 25, 30],
            "Other DL (Bytes)": [2, 4, 6, 8, 10, 12],
        }

        cls.df = pd.DataFrame(data)
        cls.analysis = UserBehaviorAnalysis(cls.df)

        def test_segment_users_by_decile(self):
            """Test the segmentation of users into deciles"""
            result = self.analysis.segment_users_by_decile()

            # Check the structure of the result
            self.assertTrue("Decile" in result.columns)
            self.assertTrue("total_data_volume" in result.columns)
            self.assertTrue("total_duration" in result.columns)

            # Check the number of rows in the result
            self.assertGreater(result.shape[0], 0)  # Ensure there are some rows

            # Check values for specific deciles
            self.assertTrue(
                result["Decile"].isin([5, 6, 7, 8, 9]).all()
            )  # Ensure only top deciles are included

        if __name__ == "__main__":
            unittest.main()
