import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os   
from prossing import TextPreprocessor
import ir_datasets

# Load the preprocessed corpus
corpus_file = r"C:\Users\Dell X360-Gen8\Desktop\preprocessed_data10.tsv"
processed_documents_df = pd.read_csv(corpus_file, delimiter='\t', header=None, names=['doc_id', 'text'])

# Drop rows with NaN values in the 'text' column (if any)
processed_documents_df = processed_documents_df.dropna(subset=['text'])

# Create a list of processed documents
processed_documents = processed_documents_df['text'].tolist()

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    min_df=10,
    max_df=0.8,
    ngram_range=(1, 2),
    smooth_idf=True,
    sublinear_tf=True
)
tfidf_matrix = vectorizer.fit_transform(processed_documents)


# Save the vectorizer
vectorizer_file = r"C:\Users\Dell X360-Gen8\.ir_datasets\trec-tot\tfidf_vectorizer2.pkl"

with open(vectorizer_file, 'wb') as file:
    joblib.dump(vectorizer, file)

# Save the TF-IDF matrix
output_file = r"C:\Users\Dell X360-Gen8\.ir_datasets\trec-tot\tfidf_matrix2.pkl"
with open(output_file, 'wb') as file:
    joblib.dump(tfidf_matrix, file)