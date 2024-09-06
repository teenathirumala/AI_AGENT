import unittest
import pandas as pd
from Data_preprocessing import process_data,clean_data
from Analysis_engine import analysis_engine
from Report_generation import generate_report

class TestDataProcessing(unittest.TestCase):
    
    def test_process_data(self):
        # Use the sample dataset provided in the assignment
        sample_data = 'olympics2024.csv'
        
        # Process the data
        processed_data = process_data(sample_data)
        
        # Check if null values have been removed
        self.assertFalse(processed_data.isnull().values.any())


class TestAnalysisEngine(unittest.TestCase):
    def setUp(self):
        self.data=pd.read_csv("olympics2024.csv")
        self.cleaned_data=clean_data(self.data)
    def test_svr(self):
        # Load the sample dataset for testing
        try:
             model, preds = analysis_engine(self.cleaned_data, algorithm='svr', target_column='Gold')
             
             print("SVR Test Passed.")
        except Exception as e:
            self.fail(f"SVR Test Failed with exception: {e}")
        # data = pd.read_csv('olympics2024.csv')
        # cleaned_data=clean_data(data)
        # # Perform linear regression
        # model,preds = self.analysis_engine(cleaned_data, 'svr', target_column='Gold')
        
        # # Test if the model makes predictions without throwing an error
        
        # self.assertEqual(len(preds), len(cleaned_data))

 
        
    def test_random_forest(self):
        # data = pd.read_csv('olympics2024.csv')
        # cleaned_data = clean_data(data)
        # model, predictions = self.analysis_engine(cleaned_data, 'Gold')
        # self.assertIsNotNone(model)
         try:
            model, preds = analysis_engine(self.cleaned_data, algorithm='random_forest', target_column='Gold')
           
            print("Random Forest Test Passed.")
         except Exception as e:
            self.fail(f"Random Forest Test Failed with exception: {e}")
            
    def test_kmeans(self):
        # Test K-Means Clustering functionality
        try:
            
            data_numeric = self.cleaned_data[['Gold', 'Silver', 'Bronze', 'Total']]  # Ensure data is numeric
            model = analysis_engine(data_numeric, algorithm='kmeans')
            self.assertIsNotNone(model)
            print("K-Means Test Passed.")
        except Exception as e:
            self.fail(f"K-Means Test Failed with exception: {e}")
            
class TestIntegration(unittest.TestCase):
    def test_full_workflow(self):
        # Load the sample dataset
        data = pd.read_csv('olympics2024.csv')
        
        # Process the data
        processed_data = process_data('olympics2024.csv')
        
        # Run analysis (linear regression)
        model = analysis_engine(processed_data, 'kmeans', target_column='Gold')
        
        # Generate report (mocked for testing purposes)
        try:
            report_generated = True
            generate_report('kmeans', model, processed_data, target_column='Gold')
            
        except Exception as e:
            report_generated = False
        
        # Assert that the report generation runs successfully
        self.assertTrue(report_generated)

if __name__ == '__main__':
    unittest.main()
