import unittest
import pandas as pd
from scripts.handset_analysis import HandsetAnalysis

class TestHandsetAnalysis(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create a sample DataFrame for testing
        data = {
            'Handset Type': ['iPhone 6S', 'iPhone 6', 'iPhone 6S', 'Galaxy S8', 'Galaxy S8', 'Galaxy A5', 'iPhone 7', 'iPhone 6', 'Galaxy J5'],
            'Handset Manufacturer': ['Apple', 'Apple', 'Apple', 'Samsung', 'Samsung', 'Samsung', 'Apple', 'Apple', 'Samsung']
        }
        df = pd.DataFrame(data)
        cls.analysis = HandsetAnalysis(df)
        
    def test_top_handsets(self):
        top_handsets = self.analysis.top_handsets(top_n=2)
        expected = pd.Series({
            'iPhone 6S': 2,
            'iPhone 6': 2
        })
        expected.index.name = 'Handset Type'
        top_handsets.index.name = 'Handset Type'
        expected.name = 'count'
        top_handsets.name = 'count'
        pd.testing.assert_series_equal(top_handsets.sort_index(), expected.sort_index())
    
    def test_top_manufacturers(self):
        top_manufacturers = self.analysis.top_manufacturers(top_n=2)
        expected = pd.Series({
            'Apple': 5,
            'Samsung': 4,
        })
        expected.index.name = 'Handset Manufacturer'
        top_manufacturers.index.name = 'Handset Manufacturer'
        expected.name = 'count'
        top_manufacturers.name = 'count'
        pd.testing.assert_series_equal(top_manufacturers.sort_index(), expected.sort_index())
    

    
    def test_top_handsets_per_manufacturer(self):
        manufacturers = ['Apple', 'Samsung']
        top_handsets = self.analysis.top_handsets_per_manufacturer(manufacturers, top_n_handsets=2)
        
        expected = {
            'Apple': pd.Series({'iPhone 6S': 2, 'iPhone 6': 2}),
            'Samsung': pd.Series({'Galaxy S8': 2, 'Galaxy A5': 1})
        }
        
        for manufacturer in manufacturers:
            expected[manufacturer].index.name = 'Handset Type'
            top_handsets[manufacturer].index.name = 'Handset Type'
            expected[manufacturer].name = 'count'
            top_handsets[manufacturer]