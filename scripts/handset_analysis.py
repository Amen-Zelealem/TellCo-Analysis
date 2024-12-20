import pandas as pd

class HandsetAnalysis:
    def __init__(self, df):
        self.df = df

    def top_handsets(self, top_n=10):
        """
        Identify the top handsets based on usage count.
        
        Parameters:
        top_n (int): Number of top handsets to return (default is 10).
        
        Returns:
        pd.Series: A Series containing the top handsets and their counts.
        """
        top_handsets = self.df["Handset Type"].value_counts().head(top_n)
        return top_handsets

    def top_manufacturers(self, top_n=3):
        """
        Identify the top handset manufacturers based on usage count.
        
        Parameters:
        top_n (int): Number of top manufacturers to return (default is 3).
        
        Returns:
        pd.Series: A Series containing the top manufacturers and their counts.
        """
        top_manufacturers = self.df["Handset Manufacturer"].value_counts().head(top_n)
        return top_manufacturers

    def top_handsets_per_manufacturer(self, manufacturers, top_n_handsets=5):
        """
        Identify the top handsets for each specified manufacturer.
        
        Parameters:
        manufacturers (list): List of manufacturers to analyze.
        top_n_handsets (int): Number of top handsets to return per manufacturer (default is 5).
        
        Returns:
        dict: A dictionary with manufacturers as keys and their top handsets as values.
        """
        results = {}
        for manufacturer in manufacturers:
            # Filter the DataFrame to include only the specified manufacturer
            df_manufacturer = self.df[self.df["Handset Manufacturer"] == manufacturer]
            # Get the top handsets for that manufacturer based on usage count
            top_handsets = (
                df_manufacturer["Handset Type"].value_counts().head(top_n_handsets)
            )
            results[manufacturer] = top_handsets
        return results