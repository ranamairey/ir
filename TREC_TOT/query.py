import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from prossing import TextPreprocessor
import ir_datasets

# Initialize the text processor
text_processor2 = TextPreprocessor()
dataset = ir_datasets.load("trec-tot/2023/train")

# Create lists to hold the document IDs and texts
doc_ids = []
texts = []

for doc in dataset.docs_iter():
    doc_ids.append(doc.doc_id)
    texts.append(doc.text)
# Load the corpus from TSV
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from prossing import TextPreprocessor
import ir_datasets

# Initialize the text processor
text_processor2 = TextPreprocessor()
dataset = ir_datasets.load("trec-tot/2023/train")

# Create lists to hold the document IDs and texts
doc_ids = []
texts = []

for doc in dataset.docs_iter():
    doc_ids.append(doc.doc_id)
    texts.append(doc.text)

# Create a DataFrame from the document IDs and texts
original_documents_df = pd.DataFrame({'doc_id': doc_ids, 'text': texts})

# Load the corpus from TSV
corpus_file = r"C:\Users\Dell X360-Gen8\Desktop\preprocessed_data5.tsv"
processed_documents_df = pd.read_csv(corpus_file, delimiter='\t', header=None, names=['doc_id', 'text'])

# Drop rows with NaN values in the 'text' column (if any)
processed_documents_df = processed_documents_df.dropna(subset=['text'])

# Create a list of processed documents
processed_documents = processed_documents_df['text'].tolist()

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    min_df=10,  # Ignore terms that appear in fewer than 10 documents
    max_df=0.8,  # Ignore terms that appear in more than 80% of the documents
    ngram_range=(1, 2),  # Consider both unigrams and bigrams
    smooth_idf=True,  # Apply smoothing to IDF weights
    sublinear_tf=True,  # Apply sublinear TF scaling
)
tfidf_matrix = vectorizer.fit_transform(processed_documents)

# Function to get similarity scores and return the full document content
def get_similarity_scores(query):
    preprocessed_query = text_processor2.preprocess_query(query)
    
    # Transform the preprocessed query into TF-IDF representation
    query_tfidf = vectorizer.transform([preprocessed_query])
    
    # Calculate cosine similarity between the query and documents
    similarity_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    
    # Get the top 10 document indices sorted by similarity score
    top_doc_indices = np.argsort(similarity_scores)[::-1][:10]
    
    # Get the IDs of the top 10 documents
    top_doc_ids = processed_documents_df.iloc[top_doc_indices]['doc_id'].values
    
    # Retrieve and print the full document content for the top 10 documents from the original dataframe
    for doc_id in top_doc_ids:
        original_doc_content = original_documents_df[original_documents_df['doc_id'] == doc_id]['text'].values[0]
        print(f"Document ID: {doc_id}")
        print(f"Text: {original_doc_content}")
        print()

# Main loop to get user query and return results
while True:
    user_query = input("Enter your query (or type 'exit' to quit): ")
    if user_query.lower() == 'exit':
        break
    
    get_similarity_scores(user_query)


# Function to get similarity scores and return the full document content
def get_similarity_scores(query):
    preprocessed_query = text_processor2.preprocess_query(query)
    
    # Transform the preprocessed query into TF-IDF representation
    query_tfidf = vectorizer.transform([preprocessed_query])
    
    # Calculate cosine similarity between the query and documents
    similarity_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    
    # Get the top 10 document indices sorted by similarity score
    top_doc_indices = np.argsort(similarity_scores)[::-1][:10]
    
    # Get the IDs of the top 10 documents
    top_doc_ids = documents_df.iloc[top_doc_indices]['doc_id'].values
    
    # Retrieve and print the full document content for the top 10 documents
    top_documents = documents_df[documents_df['doc_id'].isin(top_doc_ids)]
    for doc_id, doc_content in zip(top_documents['doc_id'], top_documents['text']):
        print(f"Document ID: {doc_id}")

        print()

while True:
    user_query = input("Enter your query (or type 'exit' to quit): ")
    if user_query.lower() == 'exit':
        break
    
    get_similarity_scores(user_query)
