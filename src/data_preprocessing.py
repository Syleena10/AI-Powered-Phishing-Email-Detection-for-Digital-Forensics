import pandas as pd
import re
import string

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)

    df = df[['Email Type', 'Email Text']]
    df.columns = ['label', 'text']

    df['label'] = df['label'].map({
        'Safe Email': 0,
        'Phishing Email': 1
    })

    df.dropna(inplace=True)

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text

    df['text'] = df['text'].apply(clean_text)

    return df
