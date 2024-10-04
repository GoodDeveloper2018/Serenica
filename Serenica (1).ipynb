import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load the data
conversations = pd.read_csv(r"file_path")

# Preprocess function to clean and preprocess the text
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_words = [word for word in word_tokens if word not in stop_words]
    
    # Lemmatize words (optional)
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    
    # Join the cleaned words back into a single string
    return ' '.join(lemmatized_words)

# Apply preprocessing to the prompt column
conversations['cleaned_prompt'] = conversations['prompt'].fillna('').apply(preprocess_text)

# Initialize vectorizer and fit on the preprocessed prompts
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(conversations["cleaned_prompt"])

# Function to get the best response
def get_best_response(user_input, df, tfidf_matrix, vectorizer):
    # Preprocess user input
    cleaned_input = preprocess_text(user_input)
    
    # Transform user input into vector
    user_input_vector = vectorizer.transform([cleaned_input])
    
    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(user_input_vector, tfidf_matrix)
    
    # Get index of the best match
    best_match_index = cosine_similarities.argmax()
    
    # Check if the best match is actually similar (optional threshold)
    if cosine_similarities[0][best_match_index] < 0.1:  # Adjust threshold as needed
        return "I'm sorry, I don't understand your question."
    
    # Return the best matched response
    return df.iloc[best_match_index]["completion"]

# Run the chatbot
while True:
    user_input = input("Ask me something: ")
    if user_input.lower() == 'exit':
        break
    response = get_best_response(user_input, conversations, tfidf_matrix, vectorizer)
    print(response)
