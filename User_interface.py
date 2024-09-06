#4)command-line interface and NLP handling
import sys
from Report_generation import generate_summary,generate_report,plot_kmeans_clusters,plot_random_forest_regression,plot_svr
from Analysis_engine import analysis_engine
from Data_preprocessing import process_data


import spacy

# Load English tokenizer, POS tagger, parser, and NER
nlp = spacy.load("en_core_web_sm")

def process_nlp_query(query, model, data, target_column):
    #function to  process natural language queries and return responses.
    query=query.lower()
    doc = nlp(query)
    
    # Simple keyword-based detection
    
    if 'cluster' in query:
        plot_kmeans_clusters(data, model)
        return "Displaying the K-Means clusters."
    elif 'finding' in query or 'result' in query:
        if target_column:
         return f"The key finding from the analysis is that the model achieved an R^2 score of {model.score(data.drop(columns=[target_column]), data[target_column]):.4f}."
        else:
            return "Findings not available without target column."
    else:
        return "Sorry, I didn't understand the question. Please try asking about the summary, clusters, or key findings."



def cli_main(cleaned_data):
    if cleaned_data.empty:
        print("Error: Cleaned data is empty.")
        return
    print("Welcome to the AI Employee CLI!")
    print("Choose an analysis type:")
    print("1. SVR")
    print("2. K-Means Clustering")
    print("3. Random Forest Regression")
    
    choice = input("Enter your choice (1/2/3): ")
    target_column=None
    model=None
    preds=None
    if choice == '1':
        target_column = input("Enter the target column for SVR ")
        model,preds = analysis_engine(cleaned_data, algorithm='svr', target_column=target_column)
        generate_report('svr', model, cleaned_data, target_column=target_column)
    elif choice == '2':
        n_clusters=int(input("Enter the number of clusters for K-Means: "))
        print(n_clusters)
        model = analysis_engine(cleaned_data,'kmeans',n_clusters=n_clusters)
        generate_report('kmeans', model, cleaned_data)
    elif choice == '3':
        target_column = input("Enter the target column for Random forest regression: ")
        model,preds = analysis_engine(cleaned_data, algorithm='randomforest', target_column=target_column)
        
        generate_report('randomforest', model, cleaned_data, target_column=target_column)
    else:
        print("Invalid choice, please restart the program.")
        sys.exit()
    
    while True:
        query = input("Ask a question about the analysis (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        response = process_nlp_query(query, model, cleaned_data, target_column)
        print(response)


