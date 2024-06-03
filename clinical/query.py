import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ir_datasets
from prossing import TextPreprocessor
from sklearn.metrics import average_precision_score
import joblib

# Initialize the text processor
text_processor = TextPreprocessor()

# Load the TREC dataset
dataset = ir_datasets.load("clinicaltrials/2017/trec-pm-2017")

# Load the preprocessed corpus
corpus_file = r"E:\preprocessed_data2.tsv"
processed_documents_df = pd.read_csv(corpus_file, delimiter='\t', header=None, names=['doc_id', 'text'])

# Drop rows with NaN values in the 'text' column (if any)
processed_documents_df = processed_documents_df.dropna(subset=['text'])

# Create a list of document IDs
doc_ids = processed_documents_df['doc_id'].tolist()

# Load vectorizer and TF-IDF matrix

vectorizer_file = r"E:\tfidf_vectorizer12.pkl"
with open(vectorizer_file, 'rb') as file:
    vectorizer = joblib.load(file)

input_file = r"E:\tfidf_matrix12.pkl"
with open(input_file, 'rb') as file:
    tfidf_matrix = joblib.load(file)

# Load qrels and convert to DataFrame
qrels = list(dataset.qrels_iter())
qrels_df = pd.DataFrame(qrels)

# Lists to store the evaluation metrics
average_precisions = []
recalls_at_10 = []
precisions_at_10 = []
reciprocal_ranks = []

# Iterate through each query in the dataset
for query in dataset.queries_iter():
    query_id = query.query_id
    query_disease = query.disease
    query_gene = query.gene
    query_demographic = query.demographic
    query_other = query.other
    modified_search_string = f"{query_id} {query_disease} {query_gene} {query_demographic} {query_other}"
    processed_query = text_processor.preprocess_text(modified_search_string)

    # Transform the query into a TF-IDF vector
    query_tfidf_vector = vectorizer.transform([processed_query])

    # Compute cosine similarity between the query and all documents
    query_cosine_similarities = cosine_similarity(query_tfidf_vector, tfidf_matrix).flatten()

    # Get the relevance judgments for the current query
    relevant_docs = qrels_df[(qrels_df['query_id'] == query_id) & (qrels_df['relevance'] > 0)]['doc_id'].tolist()

    # Debug: print relevant docs for each query
    print(f"Query ID: {query_id}, Relevant Docs: {relevant_docs}")

    if not relevant_docs:
        continue

    # Sort the documents by their similarity scores in descending order
    sorted_doc_indices = np.argsort(query_cosine_similarities)[::-1][:10]
    
    # Get the sorted document IDs for the top 10 results
    sorted_doc_ids = [doc_ids[idx] for idx in sorted_doc_indices]

    # Get the relevance judgments for the top 10 documents
    y_true = np.array([doc_id in relevant_docs for doc_id in sorted_doc_ids])  # True relevance labels
    y_score = query_cosine_similarities[sorted_doc_indices]  # Predicted relevance scores

    # Check if there are any positive examples in y_true
    if y_true.sum() == 0:
        continue

    # Calculate Average Precision for the query
    ap = average_precision_score(y_true, y_score)
    average_precisions.append(ap)

    # Calculate Recall@10
    recall_at_10 = y_true.sum() / len(relevant_docs)
    recalls_at_10.append(recall_at_10)

    # Calculate Precision@10
    precision_at_10 = y_true.sum() / 10
    precisions_at_10.append(precision_at_10)

    # Calculate Reciprocal Rank (RR)
    if y_true.sum() > 0:
        first_relevant_rank = np.where(y_true == 1)[0][0] + 1  # Rank is 1-based
        rr = 1 / first_relevant_rank
        reciprocal_ranks.append(rr)

# Calculate Mean Average Precision (MAP)
mean_average_precision = np.mean(average_precisions) if average_precisions else 0.0

# Calculate Mean Recall@10
mean_recall_at_10 = np.mean(recalls_at_10) if recalls_at_10 else 0.0

# Calculate Mean Precision@10
mean_precision_at_10 = np.mean(precisions_at_10) if precisions_at_10 else 0.0
# Calculate Mean Reciprocal Rank (MRR)
mean_reciprocal_rank = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

print(f"Mean Average Precision (MAP): {mean_average_precision}")
print(f"Mean Recall@10: {mean_recall_at_10}")
print(f"Mean Precision@10: {mean_precision_at_10}")
print(f"Mean Reciprocal Rank (MRR): {mean_reciprocal_rank}")