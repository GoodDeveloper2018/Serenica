import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import pandas as pd

# Preprocessing and sentiment analysis
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_words = [word for word in word_tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    return ' '.join(lemmatized_words)

# Sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

def get_sentiment_analysis(text):
    result = sentiment_analyzer(text)
    return result[0]['label'], result[0]['score']

# Load your dataset (Replace 'file_path' with your actual dataset path)
conversations = pd.read_csv('file_path')
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(conversations['cleaned_prompt'])

# Main chatbot function
def get_best_response(user_input, conversation_history):
    sentiment_label, confidence_score = get_sentiment_analysis(user_input)

    # Preprocess input
    cleaned_input = preprocess_text(user_input)
    user_input_vector = vectorizer.transform([cleaned_input])
    cosine_similarities = cosine_similarity(user_input_vector, tfidf_matrix)
    
    threshold = 0.15
    top_indices = np.argsort(cosine_similarities[0])[-3:]
    best_match_index = top_indices[-1]
    
    if cosine_similarities[0][best_match_index] < threshold:
        return "I'm sorry, I don't understand your question."

    response = conversations.iloc[best_match_index]["completion"]
    
    if sentiment_label == 'NEGATIVE':
        return f"It sounds like you're going through a tough time. {response}"
    elif sentiment_label == 'POSITIVE':
        return f"I'm glad to hear that! {response}"
    else:
        return response
