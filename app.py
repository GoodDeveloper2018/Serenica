from flask import Flask, render_template, request
import ml_model  # This is your ML model file

app = Flask(__name__)

# Route for the home page (GET request)
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle chatbot response (POST request)
@app.route('/get_response', methods=['POST'])  # Ensure POST is allowed here
def get_response():
    user_input = request.form['user_input']
    conversation_history = []  # Initialize conversation history (can make this session-based later)
    
    # Call the chatbot model to get a response
    response = ml_model.get_best_response(user_input, conversation_history)
    
    # Render the same page but now with user input and bot response
    return render_template('index.html', user_input=user_input, bot_response=response)

if __name__ == "__main__":
    app.run(debug=True)
