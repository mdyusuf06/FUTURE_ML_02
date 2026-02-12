import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

TEXT_COL = "Ticket Description"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def load_data(filepath):
    df = pd.read_csv(filepath)

    print("\nDataset loaded")
    print("Columns:", df.columns)

    df['clean_text'] = df[TEXT_COL].astype(str).apply(clean_text)
    df = df[df['clean_text'].str.strip() != ""]

    print("Total usable rows:", len(df))
    return df
