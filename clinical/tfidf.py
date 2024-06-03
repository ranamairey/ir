import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os   
from prossing import TextPreprocessor
import ir_datasets

# Load the preprocessed corpus

corpus_file = r"E:\preprocessed_data2.tsv"
processed_documents_df = pd.read_csv(corpus_file, delimiter='\t', header=None, names=['doc_id', 'text'])

# Drop rows with NaN values in the 'text' column (if any)
processed_documents_df = processed_documents_df.dropna(subset=['text'])

# Create a list of processed documents
processed_documents = processed_documents_df['text'].tolist()
print("read data done")
# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
   min_df=10,
   max_df=0.8,
   ngram_range=(1,2),
   smooth_idf=True,
   sublinear_tf=True,

)
print("vector done")
# Save the vectorizer
tfidf_matrix=vectorizer.fit_transform(processed_documents)

# Assuming 'tfidf_matrix' and 'vectorizer' are already created
output_file = r"E:\tfidf_matrix12.pkl"
vectorizer_file = r"E:\tfidf_vectorizer12.pkl"

# Save the TF-IDF matrix
with open(output_file, 'wb') as file:
    joblib.dump(tfidf_matrix, file)

# Save the TfidfVectorizer
with open(vectorizer_file, 'wb') as file:
    joblib.dump(vectorizer, file)