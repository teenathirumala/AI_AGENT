#1)To handle data ingestion and cleaning

import pandas as pd

def load_data(file_path):
    
    #function to Load data from CSV, JSON, or Excel files based on file extension.it takes file path as an argument and returns the loaded data as a pandas DataFrame. 
    try:
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            data = pd.read_json(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            data = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV, JSON, or Excel file.")
        
        print(f"Data loaded successfully from {file_path}")
        return data

    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def clean_data(data):
   
    #function to Perform data cleaning on the loaded DataFrame.takes the loaded dataframe as an argument, and returns the cleaned dataframe.
    try:
        # Remove duplicate rows
        data = data.drop_duplicates()
        
        #Handle missing values (example: filling NaNs with 0)
        data = data.fillna(0)
        
        # Convert columns to appropriate data types (customize as needed)
        # Example: Ensure numeric columns are of correct type
        for col in data.columns:
            if data[col].dtype == 'object':
                try:
                    data[col] = pd.to_numeric(data[col])
                except ValueError:
                    pass  # Leave as string if conversion fails

        print("Data cleaning complete.")
        return data

    except Exception as e:
        print(f"Error during data cleaning: {e}")
        return None


def process_data(file_path):

    #function to wrap up Loading and cleaning data functions from a given file path.returns he processed and cleaned DataFrame.
    data = load_data(file_path)
    if data is not None:
        data = clean_data(data)
        for column in data.select_dtypes(include=['object']).columns:
            data=pd.get_dummies(data,columns=[column],drop_first=True)
    return data
