import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import nltk
nltk.download('punkt')
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# nltk.download()    
nltk.download('punkt', download_dir='C:/Users/HP/Desktop/Server/nltk_data')
nltk.download('stopwords', download_dir='C:/Users/HP/Desktop/Server/nltk_data')

os.environ['NLTK_DATA'] = 'C:/Users/HP/Desktop/Server/nltk_data'


class TextData(BaseModel):
    data: str


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,  
    allow_methods=["*"],
    allow_headers=["*"],
)
model = joblib.load('NBmodel.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.get("/")
async def read_root():
    return {
        "message": "Welcome to the Text Categorization and Summarization API!",
        "endpoints": {
            "/": "GET - This message",
            "/predict": "POST - Submit text data for categorization and summarization",
        },
        "description": "This API allows you to categorize text into predefined categories and summarize it using LSA.",
        "model_info": {
            "model_type": "Naive Bayes",
            "vectorizer": "TF-IDF",
        }
    }

@app.post("/predict")
async def predict(data: dict):
    try:
        result = pipeline(data['data'])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8000)



def pipeline(text):
    p = heading(text)
    df_filtered = categorizing(p)
    df_filtered.loc[:, 'cleaned_text'] = df_filtered['Text'].apply(preprocess_text)
    df_filtered = df_filtered.dropna(subset=['cleaned_text'])
    df_filtered = df_filtered[df_filtered['cleaned_text'].str.strip() != '']
    policy_tfidf = vectorizer.transform(df_filtered['cleaned_text'])
    df_filtered['predicted_category'] = model.predict(policy_tfidf)
    summaries = summarize_by_category(df_filtered, 'Text', 'predicted_category', 1)

    result = {}
    for category, summary in summaries.items():
        result[category] = summary  


    return result


def heading(text):
    paragraphs = text.strip().split('\n\n')

    first_sentences = []
    rest_sentences = []

    for paragraph in paragraphs:
        sentences = nltk.sent_tokenize(paragraph)
        valid_first_sentence = ''
        rest_of_sentences = ''
        for i, sentence in enumerate(sentences):
            if any(word.isalpha() for word in sentence.split()):
                valid_first_sentence = sentence
                rest_of_sentences = '. '.join(sentences[:i] + sentences[i+1:])
                break
        if not valid_first_sentence and sentences:
            valid_first_sentence = sentences[-1]
            rest_of_sentences = '. '.join(sentences[:-1])

        first_sentences.append(valid_first_sentence)
        rest_sentences.append(rest_of_sentences)
        
    duplicated_categories = []
    all_sentences = []

    # Iterate over paragraphs
    for paragraph_index in range(len(first_sentences)):
        num_sentences = len(rest_sentences[paragraph_index].split('. '))
        category = first_sentences[paragraph_index]
        duplicated_categories.extend([category] * num_sentences)
        all_sentences.extend(rest_sentences[paragraph_index].split('. '))


    df = pd.DataFrame({'Categories': duplicated_categories, 'Text': all_sentences}) 

    return df    


def categorizing(df):
    for index, row in df.iterrows():
        if "collect" in row['Categories'] or "Collect" in row['Categories']:
            df.at[index, 'Categories'] = "Data Collection"
    for index, row in df.iterrows():
        if "share" in row['Categories'] or "Share" in row['Categories'] or "Sharing" in row ['Categories']:
            df.at[index, 'Categories'] = "Data Sharing"    
    for index, row in df.iterrows():
        if "Rights" in row['Categories'] or "rights" in row['Categories'] or "obligations" in row ['Categories']:
            df.at[index, 'Categories'] = "Rights and Protection"
    for index, row in df.iterrows():
        if "Use" in row['Categories'] or "Usage" in row['Categories'] or "use" in row ['Categories']:
            df.at[index, 'Categories'] = "Data Usage"       
    for index, row in df.iterrows():
        if "Store" in row['Categories'] or "Storage" in row['Categories'] or "store" in row ['Categories']:
            df.at[index, 'Categories'] = "Data Storage"         
    for index, row in df.iterrows():
        if "keep" in row['Categories']:
            df.at[index, 'Categories'] = "Data Storage"    
    for index, row in df.iterrows():
        if "with" in row['Categories']:
            df.at[index, 'Categories'] = "Data Usage" 
    for index, row in df.iterrows():
        if "need" in row['Categories']:
            df.at[index, 'Categories'] = "Data Collection"     
    for index, row in df.iterrows():
        if "Affiliated" in row['Categories']:
            df.at[index, 'Categories'] = "Data Sharing"                
    for index, row in df.iterrows():
        if "Managing" in row['Categories']:
            df.at[index, 'Categories'] = "Rights and Protection"

    valid_categories = ["Data Collection", "Data Sharing", "Data Storage", "Data Usage", "Rights and Protection"]

    
    df_filtered = df[df['Categories'].isin(valid_categories)]        

    return df_filtered       

def preprocess_text(text):
    # Tokenization
    stop_words = list(STOP_WORDS)
    tokens = word_tokenize(text.lower())

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)





def lsa_summarize(df_filtered, text_column, num_sentences):
    # Split text into individual sentences
    text = ' '.join(df_filtered[text_column].dropna())
    
    # Split text into individual sentences
    sentences = [sent.strip() for sent in text.split('.') if sent.strip()]
    
    # Check if there are enough sentences to summarize
    if len(sentences) <= num_sentences:
        return '. '.join(sentences)  # Return the entire text if there are not enough sentences
    
    # Create CountVectorizer and TfidfTransformer
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    transformer = TfidfTransformer()
    X_tfidf = transformer.fit_transform(X)
    
    # Apply LSA (TruncatedSVD)
    svd = TruncatedSVD(n_components=min(num_sentences, X_tfidf.shape[0]-1))
    X_svd = svd.fit_transform(X_tfidf)
    
    # Determine top sentences based on the highest LSA scores
    top_sentence_indices = np.argsort(X_svd[:, 0])[::-1][:num_sentences]
    
    # Sort indices to maintain the original order
    top_sentence_indices = sorted(top_sentence_indices)
    
    # Generate the summary
    summary = '. '.join([sentences[i] for i in top_sentence_indices])
    
    return summary


def summarize_by_category(df, text_column, category_column, num_sentences):
    # Create a dictionary to store the summaries for each category
    summaries = {}

    # Get unique categories
    categories = df[category_column].unique()
    
    for category in categories:
        # Filter texts belonging to the current category
        df_filtered = df[df[category_column] == category]
        
        # Apply the LSA summarizer to the filtered texts
        summary = lsa_summarize(df_filtered, text_column, num_sentences)
        
        # Store the summary in the dictionary
        summaries[category] = summary
    
    return summaries

