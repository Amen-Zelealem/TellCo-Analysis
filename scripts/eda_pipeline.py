import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class EDA:
    
    def __init__(self, df):
        self.df = df

    def compute_basic_metrics(self, variables):
        metrics = {}
        
        # Columns of interest
        # columns_of_interest = ['Dur. (ms)', 'Total UL (Bytes)', 'Total DL (Bytes)']
        
        for var in variables:
            metrics[var] = {
                'Mean': self.df[var].mean(),
                'Median': self.df[var].median(),
                'Mode': self.df[var].mode().values[0],  # mode() returns a Series
                'Standard Deviation': self.df[var].std(),
                'Variance': self.df[var].var(),
                'Range': self.df[var].max() - self.df[var].min(),
                'IQR': self.df[var].quantile(0.75) - self.df[var].quantile(0.25)
    }

        
        # Convert the metrics into a DataFrame for better presentation
        metrics_df = pd.DataFrame(metrics)
        return metrics_df
      

    def plot_distribution(self, column):
        sns.histplot(self.df[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.show()

    def plot_correlation_matrix(self):
        corr_matrix = self.df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()
        
    def univariate_analysis(self):
        # Example: Plot distribution of session duration
        plt.figure(figsize=(13, 6))
        sns.histplot(self.df['Session Duration'], bins=50, kde=True,palette='pastel')
        plt.title('Distribution of Session Duration')
        plt.show()

    def bivariate_analysis(self):
        # Example: Scatter plot of YouTube Data vs. Total Data
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.df['Youtube DL (Bytes)'] + self.df['Youtube UL (Bytes)'], y=self.df['Total DL (Bytes)'] + self.df['Total UL (Bytes)'])
        plt.title('YouTube Data vs. Total Data')
        plt.show()
        
    def compute_correlation_matrix(self, variables):
        """
        Compute a correlation matrix for specified variables.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing data.
        
        Returns:
        pd.DataFrame: Correlation matrix of the specified variables.
        """
       
        # Aggregate total data for each variable
        for var in variables:
            if var not in self.df.columns:
                raise KeyError(f"Column '{var}' not found in DataFrame.")
        
        # Calculate total data for each application
        for app in ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']:
            self.df[f'{app} Total (Bytes)'] = (
                self.df[f'{app} UL (Bytes)'] + self.df[f'{app} DL (Bytes)']
            )
        
        # Select relevant columns for correlation matrix
        relevant_columns = [f'{app} Total (Bytes)' for app in ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']]
        
        # Compute correlation matrix
        correlation_matrix = self.df[relevant_columns].corr()
        
        return correlation_matrix
    
    def perform_pca(self, variables):
        """
        Perform Principal Component Analysis (PCA) on the dataset.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing data.
        
        Returns:
        pd.DataFrame: DataFrame with PCA results.
        """
        # Prepare the data
        data = self.df[variables].dropna()  # Drop rows with missing values
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Apply PCA
        pca = PCA(n_components=2)  # Reduce to 2 components for easy interpretation
        pca_result = pca.fit_transform(data_scaled)
        
        # Create a DataFrame with PCA results
        pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
        
        # Variance explained by each principal component
        explained_variance = pca.explained_variance_ratio_
        
        return pca_df, explained_variance