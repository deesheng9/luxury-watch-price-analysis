import unittest
import pandas as pd
import sys
import os

# Ensure Python can find the preprocessing directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.cleaner import clean_data

class TestCleaner(unittest.TestCase):
    def test_clean_data(self):
        """
        Tests whether the clean_data function correctly handles missing values 
        and cleans price symbols.
        """
        print("\nRunning Test: test_clean_data...")
        
        # 1. Prepare mock data
        mock_data = {
            'Brand': ['Rolex', 'Omega', 'Seiko', None], # Intentionally include a missing value (None)
            'Price (USD)': ['$10,000', '$5,500', '200', '$1,000'] # Intentionally include '$' and commas
        }
        df = pd.DataFrame(mock_data)

        # 2. Execute the cleaning function
        cleaned_df = clean_data(df)

        # 3. Assertions (Crucial for validation)
        
        # Test A: Check if the row with the missing value was successfully dropped
        # Originally 4 rows; it should be 3 after dropping the null value
        self.assertEqual(len(cleaned_df), 3, "Missing values were not dropped correctly!")

        # Test B: Check if the price was successfully converted from string to float
        self.assertEqual(cleaned_df['Price (USD)'].iloc[0], 10000.0, "Price string was not converted to float correctly!")
        
        # Test C: Check if the data type of the price column is float
        self.assertTrue(pd.api.types.is_float_dtype(cleaned_df['Price (USD)']), "Price column is not numeric!")

        print("Test Passed: Data cleaning works perfectly!")

if __name__ == '__main__':
    unittest.main()