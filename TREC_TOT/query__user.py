import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from prossing import TextPreprocessor
import ir_datasets
import nltk
from nltk.corpus import wordnet
import string
from nltk.corpus import stopwords

# Initialize the text processor
text_processor2 = TextPreprocessor()
dataset = ir_datasets.load("trec-tot/2023/train")

# Create empty lists to hold document IDs and texts
doc_ids = []
titles = []
texts = []

# Loop through the dataset and append the information to the lists
for doc in dataset.docs_iter():
    doc_ids.append(doc.doc_id)
    titles.append(doc.page_title)
    texts.append(doc.text)
    # Assuming you have access to Wikidata IDs and classes

# Create a DataFrame from the extracted document information
original_documents_df = pd.DataFrame({'doc_id': doc_ids, 'title': titles, 'texts': texts})

# Initialize a TfidfVectorizer object with specific parameters
vectorizer_file = r"C:\Users\Dell X360-Gen8\.ir_datasets\trec-tot\tfidf_vectorizer2.pkl"
with open(vectorizer_file, 'rb') as file:
    vectorizer = joblib.load(file)

# Transform the processed documents into a TF-IDF matrix
input_file = r"C:\Users\Dell X360-Gen8\.ir_datasets\trec-tot\tfidf_matrix2.pkl"
with open(input_file, 'rb') as file:
    tfidf_matrix = joblib.load(file)

def preprocess_query_with_suggestions(query):
    """
    Preprocesses and suggests alternative queries using synonyms,
    replacing only one keyword at a time.

    Args:
        query (str): The user's search query.

    Returns:
        str: The processed query or a suggested query.
    """
    # 1. Extract Keywords from the Query
    
    stop_words = set(stopwords.words('english'))
    punctuation_symbols = set(string.punctuation)
    keywords1 = [word for word in query.split() if word.lower() not in stop_words and word not in punctuation_symbols]

    # 2. Find Synonyms for Each Keyword
    synonyms = {}
    for keyword in keywords1:
        synsets = wordnet.synsets(keyword)
        if synsets:  # Check if synonyms exist for the keyword
            for synset in synsets:
                for lemma in synset.lemmas():
                    synonym_word = lemma.name().lower()  # Lowercase for case-insensitive matching
                    if synonym_word not in query.lower().split() and synonym_word not in stop_words and synonym_word not in punctuation_symbols:
                        if keyword not in synonyms:
                            synonyms[keyword] = set()
                        synonyms[keyword].add(synonym_word)

    # 3. Generate Unique Query Suggestions
    query_suggestions = set()
    for keyword, synonym_set in synonyms.items():
        for synonym in synonym_set:
            query_suggestions.add(f"{keyword} OR {synonym}")

    # 4. Display Query Suggestions (if any)
    if query_suggestions:
        print("Query suggestions:")
        suggestion_count = 1
        suggestion_list = list(query_suggestions)
        for suggestion in suggestion_list:
            print(f"{suggestion_count}. {suggestion}")
            suggestion_count += 1

        # 5. Prompt User for Input and Handle Choices
        modified_query = query  # Start with the original query
        for keyword, synonym_set in synonyms.items():
            print(f"Choose a synonym for '{keyword}':")
            for i, synonym in enumerate(synonym_set, 1):
                print(f"{i}. {synonym}")
            user_choice = input("Enter the number of your choice, or 'r' to refine your query: ")
            if user_choice.lower() == 'r':
                return query  # Return original query if refinement is chosen
            try:
                choice_num = int(user_choice)
                if 1 <= choice_num <= len(synonym_set):
                    chosen_synonym = list(synonym_set)[choice_num - 1]
                    modified_query = modified_query.replace(keyword, chosen_synonym)  # Replace the chosen synonym in the query
                else:
                    print(f"Invalid choice number. Please try again.")
                    return query  # Return original query if invalid choice
            except ValueError:
                print(f"Invalid input. Please enter the number of your choice.")
                return query  # Return original query if invalid input
        return modified_query

    # 6. No Suggestions or User Chose Refinement
    return query

def get_similarity_scores(query, with_query_refinement=True):
    """
    Gets similarity scores and document content based on a user query.

    Args:
        query (str): The user's search query.
        with_query_refinement (bool, optional): Flag to enable query refinement. Defaults to True.

    Returns:
        None
    """
    query_suggestions = preprocess_query_with_suggestions(query)

    print(query_suggestions)
  # Preprocess the chosen query (original or suggested) using your custom logic
    preprocessed_query = text_processor2.preprocess_query(query_suggestions)

    # Transform the preprocessed query into TF-IDF representation
    query_tfidf = vectorizer.transform([preprocessed_query])

    # Calculate cosine similarity between the query and documents
    similarity_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    # Get the top 10 document indices sorted by similarity score
    top_doc_indices = np.argsort(similarity_scores)[::-1][:10]

    # Get the IDs of the top 10 documents
    top_doc_ids = original_documents_df.iloc[top_doc_indices]['doc_id'].values

    if not top_doc_ids.any():  # Check if there are any retrieved documents
        print("No documents found matching your query.")
        return  # Exit the function if no documents found

    # Retrieve and print content for the top 10 documents
    for doc_id in top_doc_ids:
        original_doc_content = original_documents_df[original_documents_df['doc_id'] == doc_id]
        print(f"Document ID: {doc.doc_id}")
        print(f"Title: {doc.page_title}")
        print(f"Text: {doc.text}")
        print()

# Main loop for user interaction
while True:
    # Request user query
    user_query = input("Enter your search query (or 'q' to quit): ")

    if user_query.lower() == 'q':
        break

    # Call the function to get similarity scores and display results
    get_similarity_scores(user_query)

print("Exiting the program...")
