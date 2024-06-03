from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity
from prossing import TextPreprocessor
import ir_datasets
import joblib
import os

app = Flask(__name__)

# Initialize the text processor
text_processor2 = TextPreprocessor()
dataset = ir_datasets.load("clinicaltrials/2017/trec-pm-2017")
# Create lists to hold the document IDs and texts
# Create lists to hold the document IDs and texts
doc_ids = []
eligibility = []
detailed_description=[]
summary=[]
condition=[]
title=[]
for doc in dataset.docs_iter():
    doc_ids.append(doc.doc_id)
    eligibility.append(doc.eligibility)
    detailed_description.append(doc.detailed_description)
    summary.append(doc.summary)
    condition.append(doc.condition)
    title.append(doc.title)

# Create a DataFrame from the collected information
original_documents_df = pd.DataFrame({'doc_id': doc_ids, 'title': title,'condition':condition,'summary':summary,'detailed_description':detailed_description,'eligibility':eligibility})


# Load the saved TF-IDF matrix and vectorizer
save_dir = r"E:"
tfidf_matrix_file = os.path.join(save_dir, 'tfidf_matrix12.pkl')
vectorizer_file = os.path.join(save_dir, 'tfidf_vectorizer12.pkl')

tfidf_matrix = joblib.load(tfidf_matrix_file)
vectorizer = joblib.load(vectorizer_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    preprocessed_query = text_processor2.preprocess_query(query)
    query_tfidf = vectorizer.transform([preprocessed_query])
    similarity_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[::-1][:10]
    top_docs = original_documents_df.iloc[top_indices]
    return render_template('results.html', query=query, results=top_docs)

if __name__ == '__main__':
  app.run(debug=True, port=8000)
