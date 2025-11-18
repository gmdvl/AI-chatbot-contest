import os
from bert import EnhancedSTEMTutorBot

# --------------------------------------------------------------------------
# 1. FLASK APP SETUP & CHATBOT LOGIC
# --------------------------------------------------------------------------

try:
    # Attempt to import all dependencies required by the user's chatbot
    from flask import Flask, request, jsonify, render_template_string
except ImportError as e:
    print(f"FATAL ERROR: Missing required library: {e}")
    print("Please run: pip install Flask transformers torch datasets sentence-transformers numpy")


# --------------------------------------------------------------------------
# 2. FLASK APPLICATION AND BOT INSTANCE
# --------------------------------------------------------------------------

app = Flask(__name__)

# Initialize the Bot Globally
STEM_BOT = EnhancedSTEMTutorBot()

# --------------------------------------------------------------------------
# 3. EMBEDDED HTML/CSS/JS FRONTEND
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(BASE_DIR, "templates", "index.html")


with open(HTML_PATH, "r", encoding="utf-8") as f:
    HTML_TEMPLATE = f.read()

@app.route("/")
def home():
    return HTML_TEMPLATE

# --------------------------------------------------------------------------
# 4. FLASK ROUTES
# --------------------------------------------------------------------------

@app.route('/')
def index():
    """Route to serve the main chat interface."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/chat', methods=['POST'])
def chat():
    """API route to receive user messages and send back chatbot responses."""
    try:
        data = request.get_json()
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        response_dict = STEM_BOT.chat(user_message)
        
        return jsonify({
            'response': response_dict.get('answer', "I'm still learning and don't have an answer for that yet."),
            'confidence': response_dict.get('confidence', 0.0)
        })

    except Exception as e:
        app.logger.error(f"An error occurred in the chat API: {e}")
        return jsonify({'error': 'Internal Server Error', 'response': 'A severe server error occurred while processing your request.'}), 500

# --------------------------------------------------------------------------
# 5. RUN THE APP
# --------------------------------------------------------------------------

if __name__ == '__main__':
    print("Running Flask STEM Tutor Bot. This may take a minute for model initialization...")
    app.run(debug=True, threaded=False)