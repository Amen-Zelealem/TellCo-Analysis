import pandas as pd

class UserBehaviorAnalysis:
    
    def __init__(self, df):
        self.df = df
    
    def aggregate_user_behavior(self):
        """
        Aggregate user behavior from xDR data.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing xDR data.
        
        Returns:
        pd.DataFrame: Aggregated user behavior data.
        """
        # Group the data by 'Bearer Id' to aggregate user behavior metrics
        user_behavior = self.df.groupby('Bearer Id').agg(
            num_sessions=('Bearer Id', 'count'),  # Count the number of sessions per user
            total_duration=('Dur. (ms)', 'sum'),  # Sum total session duration in milliseconds
            total_download=('Total Download (Bytes)', 'sum'),  # Sum total download data in bytes
            total_upload=('Total Upload (Bytes)', 'sum')   # Sum total upload data in bytes
        )

        # Calculate total data volume by adding download and upload totals
        user_behavior['total_data_volume'] = user_behavior['total_downloads'] + user_behavior['total_uploads']

        # Define a list of common applications to analyze
        applications = ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']

        # Add total data usage for each application
        for app in applications:
            user_behavior[f'{app} Total (Bytes)'] = (
                self.df.groupby('Bearer Id')[f'{app} UL (Bytes)'].sum() + 
                self.df.groupby('Bearer Id')[f'{app} DL (Bytes)'].sum()
            )

        return user_behavior

    def segment_users_by_decile(self):
        """
        Segment users into the top five decile classes based on session duration and compute total data per decile class.
        
        Returns:
        pd.DataFrame: DataFrame with decile class data.
        """
        # Aggregate user behavior data
        user_behavior = self.aggregate_user_behavior()
        
        # Ensure that 'total_duration' and 'total_data_volume' are numeric for calculations
        user_behavior['total_duration'] = pd.to_numeric(user_behavior['total_duration'], errors='coerce')
        user_behavior['total_data_volume'] = pd.to_numeric(user_behavior['total_data_volume'], errors='coerce')
        
        # Create deciles based on total duration, assigning labels while handling duplicates
        user_behavior['Decile'] = pd.qcut(user_behavior['total_duration'], 10, labels=False, duplicates='drop')

        # Filter to keep only the top five deciles (i.e., the users with the longest session durations)
        top_five_deciles = user_behavior[user_behavior['Decile'] >= 5]

        # Compute total data and duration for each of the top five decile classes
        decile_summary = top_five_deciles.groupby('Decile').agg({
            'total_data_volume': 'sum',  # Sum total data volume for each decile
            'total_duration': 'sum'       # Sum total duration for each decile
        }).reset_index()

        return decile_summary