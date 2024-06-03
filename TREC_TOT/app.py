from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity
from prossing import TextPreprocessor
import ir_datasets

app = Flask(__name__)

# Initialize the text processor
text_processor2 = TextPreprocessor()
dataset = ir_datasets.load("trec-tot/2023/train")

# Create lists to hold the document IDs and texts
doc_ids = []
titles = []
texts = []
wikidata_ids = []
wikidata_classes = []
sections = []
infoboxes = []

# Loop through the dataset and append the information to the lists
for doc in dataset.docs_iter():
    doc_ids.append(doc.doc_id)
    titles.append(doc.page_title)
    texts.append(doc.text)
    # Assuming you have access to Wikidata IDs and classes
    wikidata_ids.append(doc.wikidata_id)
    wikidata_classes.append(doc.wikidata_classes)
    # Assuming you have access to sections and infoboxes
    sections.append(doc.sections)
    infoboxes.append(doc.infoboxes)

# Create a DataFrame from the collected information
original_documents_df = pd.DataFrame({
    'doc_id': doc_ids,
    'title': titles,
    'text': texts,
    'wikidata_id': wikidata_ids,
    'wikidata_classes': wikidata_classes,
    'sections': sections,
    'infoboxes': infoboxes
})
# Load the preprocessed corpus
corpus_file = r"C:\Users\Dell X360-Gen8\Desktop\preprocessed_data5.tsv"
processed_documents_df = pd.read_csv(corpus_file, delimiter='\t', header=None, names=['doc_id', 'text'])

# Drop rows with NaN values in the 'text' column (if any)
processed_documents_df = processed_documents_df.dropna(subset=['text'])

# Load the saved TF-IDF matrix and vectorizer
save_dir = r"C:\Users\Dell X360-Gen8\.ir_datasets\trec-tot"
tfidf_matrix_file = os.path.join(save_dir, 'tfidf_matrix2.pkl')
vectorizer_file = os.path.join(save_dir, 'tfidf_vectorizer2.pkl')

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
    app.run(debug=True,port=8000)
