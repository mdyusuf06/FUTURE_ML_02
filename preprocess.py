import pandas as pd
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

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)

    print("\nColumns in dataset:", df.columns)


    text_column = "Ticket Description"

    df['clean_text'] = df[text_column].astype(str).apply(clean_text)


    df = df[df['clean_text'].str.strip() != ""]

    print("Total usable rows:", len(df))

    return df
