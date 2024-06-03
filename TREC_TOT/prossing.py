import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re
import inflect
from nltk.stem import WordNetLemmatizer
import ir_datasets

nlp = spacy.load('en_core_web_sm')
stemmer = PorterStemmer()

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.punctuation = string.punctuation
        self.inflect_engine = inflect.engine()

    def numbers_to_words(self, text):
        words = word_tokenize(text)
        converted_words = []
        for word in words:
            if word.isdigit():
                try:
                    converted_word = self.inflect_engine.number_to_words(word)
                    converted_words.append(converted_word)
                except inflect.NumOutOfRangeError:
                    converted_words.append("[Number Out of Range]")
            else:
                converted_words.append(word)
        return ' '.join(converted_words)

    def cleaned_text(self, text):
        cleaned_text = re.sub(r'[^\w\s]', '', text)
        return cleaned_text

    def normalization_example(self, text):
        return text.lower()
        
    def stemming_example(self, text):
        words = word_tokenize(text)
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)

    def lemmatization_example(self, text):
        words = word_tokenize(text)
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)

    def remove_stopwords(self, text):
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)


    def preprocess_text(self, text):
        cleaned_text = self.cleaned_text(text)
        normalized_text = self.normalization_example(cleaned_text)
        stemmed_text = self.stemming_example(normalized_text)
        lemmatized_text = self.lemmatization_example(stemmed_text)
        text_without_stopwords = self.remove_stopwords(lemmatized_text)
        final_text = self.numbers_to_words(text_without_stopwords)
        print('done')
        return final_text

    def preprocess_query(self, query):
        return self.preprocess_text(query)

    def preprocess_title(self, title):
        return self.preprocess_text(title)

