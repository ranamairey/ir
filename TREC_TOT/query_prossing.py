import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re
import spacy
import csv  
import inflect
import ir_datasets
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib
import jsonlines
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from prossing import TextPreprocessor  # تأكد من أن لديك هذه الوحدة

# Initialize inflect engine
p = inflect.engine()

# Load the dataset
dataset = ir_datasets.load("trec-tot/2023/train")

# Load the queries
queries = []
query_ids = []
for query in dataset.queries_iter():
    queries.append(query.text)
    query_ids.append(query.query_id)

# Create a TextPreprocessor object
text_processor1 = TextPreprocessor()

# Process all queries
preprocessed_queries = {}  # Initialize an empty dictionary for preprocessed queries with IDs
for query_id, query_text in zip(query_ids, queries):
    processed_query = text_processor1.preprocess_query(query_text)
    preprocessed_queries[query_id] = processed_query

# Save preprocessed queries to a TSV file
queries_file = r"C:\Users\Dell X360-Gen8\Desktop\preprocessed_queries1.tsv"
with open(queries_file, "w", newline='') as outfile:
    writer = csv.writer(outfile, delimiter='\t')
    for query_id, processed_query in preprocessed_queries.items():
        writer.writerow([query_id, processed_query])

queries_df = pd.read_csv(queries_file, delimiter='\t', encoding='latin1', header=None, names=['id', 'text'])


# Drop rows with NaN values in the 'text' column (if any)
queries_df = queries_df.dropna(subset=['text'])

# Create a list of queries
queries = queries_df['text'].tolist()

# Load the previously saved TfidfVectorizer
vectorizer_file = r"C:\Users\Dell X360-Gen8\.ir_datasets\trec-tot\tfidf_vectorizer2.pkl"
with open(vectorizer_file, 'rb') as file:
    vectorizer = joblib.load(file)

# Transform queries into TF-IDF representation
queries_tfidf_matrix = vectorizer.transform(queries)

# Save the TF-IDF matrix for queries
queries_tfidf_file = r"C:\Users\Dell X360-Gen8\.ir_datasets\trec-tot\queries_tfidf_matrix2.pkl"
with open(queries_tfidf_file, 'wb') as file:
    joblib.dump(queries_tfidf_matrix, file)
