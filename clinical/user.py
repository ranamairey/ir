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
dataset = ir_datasets.load("clinicaltrials/2017/trec-pm-2017")

# Create empty lists to hold document IDs and texts
doc_ids = []
eligibility = []
detailed_description = []
summary = []
condition = []
title = []
# Extract document information from the dataset (limited to 3 documents for brevity)
for doc in dataset.docs_iter():
    doc_ids.append(doc.doc_id)
    eligibility.append(doc.eligibility)
    detailed_description.append(doc.detailed_description)
    summary.append(doc.summary)
    condition.append(doc.condition)
    title.append(doc.title)

# Create a DataFrame from the extracted document information
original_documents_df = pd.DataFrame({'doc_id': doc_ids, 'title': title, 'condition': condition, 'summary': summary, 'detailed_description': detailed_description, 'eligibility': eligibility})

# Initialize a TfidfVectorizer object with specific parameters

vectorizer_file = r"E:\tfidf_vectorizer12.pkl"
with open(vectorizer_file, 'rb') as file:
    vectorizer = joblib.load(file)

input_file = r"E:\tfidf_matrix12.pkl"
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
  synonyms = []
  for keyword in keywords1:
      synsets = wordnet.synsets(keyword)
      if synsets:  # Check if synonyms exist for the keyword
          for synset in synsets:
              for lemma in synset.lemmas():
                  synonym_word = lemma.name().lower()  # Lowercase for case-insensitive matching
                  if synonym_word not in query.lower().split() and synonym_word not in stop_words and synonym_word not in punctuation_symbols:
                      synonyms.append(synonym_word)

  # 3. Generate Unique Query Suggestions
  query_suggestions = set()
  for keyword in keywords1:
      for synonym in synonyms:
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
      user_choice = input("Enter 's' followed by suggestion number(s) (e.g., s1 s3 s5) to choose, or 'r' to refine your query: ")
      if user_choice.lower().startswith("s"):
          # Modify the original query even if there's an error in user input (avoid returning the original query)
          modified_query = query
          for suggestion in user_choice.split()[1:]:
              try:
                  suggestion_num = int(suggestion[1:])
                  if 1 <= suggestion_num <= len(suggestion_list):
                      chosen_suggestion = suggestion_list[suggestion_num - 1]
                      keyword, synonym = chosen_suggestion.split(" OR ")
                      # Replace the keyword with the synonym in the original query
                      modified_query = modified_query.lower().replace(keyword, synonym)
                  else:
                      print(f"Invalid suggestion number {suggestion_num}.")
              except ValueError:
                  print(f"Invalid input {suggestion}. Please enter 's' followed by a number.")


          return modified_query
      elif user_choice.lower() == "r":
          return query
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
        print(f"Document ID: {doc_id}")
        print(f"Title: {original_doc_content['title'].values[0]}")
        print(f"Eligibility: {original_doc_content['eligibility'].values[0]}")
        print(f"Condition: {original_doc_content['condition'].values[0]}")
        print(f"Summary: {original_doc_content['summary'].values[0]}")
        print(f"Detailed Description: {original_doc_content['detailed_description'].values[0]}")
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