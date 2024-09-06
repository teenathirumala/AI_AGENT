#Entry point to run the AI employee
import pandas as pd
import sys
from Data_preprocessing import process_data
from Analysis_engine import analysis_engine
from Report_generation import generate_report
from User_interface import cli_main


def main():
#  Main function executes the AI employee workflow.
#     1. Ingest and clean the data.
#     2. Launch the command-line interface (CLI) for user interaction.
    print("Welcome to the AI Employee System!")
    
    # Get dataset path from the user
    dataset_path = input("Enter the path to your dataset (CSV/Excel/JSON): ")
    
    try:
        # Process the data (cleaning and preparation)
        cleaned_data = process_data(dataset_path)
        if cleaned_data is None or cleaned_data.empty:
            raise ValueError("Data could not be loaded or is empty.")
    except Exception as e:
        print(f"Error processing the data: {e}")
        sys.exit()

    # Start the CLI to interact with the user
    cli_main(cleaned_data)

if __name__ == "__main__":
    main()
