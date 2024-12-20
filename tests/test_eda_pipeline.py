import unittest
import pandas as pd
from scripts.eda_pipeline import EDA  # Adjust the import path as needed


class TestEDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Sample data for testing
        cls.data = pd.DataFrame(
            {
                "Session Duration": [1000, 2000, 3000, 4000, 5000],
                "Youtube DL (Bytes)": [100, 200, 300, 400, 500],
                "Youtube UL (Bytes)": [50, 100, 150, 200, 250],
                "Total DL (Bytes)": [150, 300, 450, 600, 750],
                "Total UL (Bytes)": [75, 150, 225, 300, 375],
                "Social Media UL (Bytes)": [10, 20, 30, 40, 50],
                "Social Media DL (Bytes)": [5, 10, 15, 20, 25],
                "Google UL (Bytes)": [15, 25, 35, 45, 55],
                "Google DL (Bytes)": [10, 20, 30, 40, 50],
                "Email UL (Bytes)": [20, 30, 40, 50, 60],
                "Email DL (Bytes)": [10, 20, 30, 40, 50],
                "Netflix UL (Bytes)": [5, 15, 25, 35, 45],
                "Netflix DL (Bytes)": [2, 10, 18, 28, 40],
                "Gaming UL (Bytes)": [50, 100, 150, 200, 250],
                "Gaming DL (Bytes)": [25, 50, 75, 100, 125],
                "Other UL (Bytes)": [5, 10, 15, 20, 25],
                "Other DL (Bytes)": [3, 7, 10, 15, 20],
            }
        )
        cls.eda = EDA(cls.data)

    def test_plot_distribution(self):
        try:
            self.eda.plot_distribution("Session Duration")
        except Exception as e:
            self.fail(f"Plotting distribution failed: {e}")

    def test_plot_correlation_matrix(self):
        try:
            self.eda.plot_correlation_matrix()
        except Exception as e:
            self.fail(f"Plotting correlation matrix failed: {e}")

    def test_univariate_analysis(self):
        try:
            self.eda.univariate_analysis()
        except Exception as e:
            self.fail(f"Univariate analysis plotting failed: {e}")

    def test_bivariate_analysis(self):
        try:
            self.eda.bivariate_analysis()
        except Exception as e:
            self.fail(f"Bivariate analysis plotting failed: {e}")

    def test_perform_pca(self):
        variables = ["Session Duration", "Total DL (Bytes)", "Total UL (Bytes)"]
        pca_df, explained_variance = self.eda.perform_pca(variables)
        self.assertEqual(pca_df.shape[1], 2)  # Should have 2 components
        self.assertEqual(
            len(explained_variance), 2
        )  # Variance should be explained for 2 components


if __name__ == "__main__":
    unittest.main()
