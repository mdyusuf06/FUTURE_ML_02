import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def load_models():
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    category_model = pickle.load(open("category_model.pkl", "rb"))
    priority_model = pickle.load(open("priority_model.pkl", "rb"))

    return vectorizer, category_model, priority_model

def predict_ticket(text):
    vectorizer, category_model, priority_model = load_models()

    clean = clean_text(text)
    vec = vectorizer.transform([clean])

    category = category_model.predict(vec)[0]
    priority = priority_model.predict(vec)[0]

    print("\n===== PREDICTION RESULT =====")
    print("Ticket:", text)
    print("Predicted Category:", category)
    print("Predicted Priority:", priority)
