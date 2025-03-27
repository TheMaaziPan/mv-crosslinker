import re
import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from tqdm import tqdm

# Initialize NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_string_and_remove_stopwords(text):
    """Clean product titles for keyword generation"""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        words = [lemmatizer.lemmatize(word) for word in text.split() 
                if word not in stop_words and len(word) > 2]
        return ' '.join(words)
    except:
        return text.lower()

def get_similarity(text1, text2):
    """Calculate similarity score (0-10 scale)"""
    if not text1 or not text2:
        return 0
    try:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([text1, text2])
        return round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 10, 2)
    except:
        return 0

def generate_mock_volumes(keywords, months=None):
    """Generate mock search volumes for New Look products"""
    np.random.seed(42)  # Consistent results
    data = {
        'keyword': keywords,
        'avg_monthly_searches': np.random.randint(100, 5000, len(keywords))
    }
    if months:
        for month in months:
            data[month] = np.random.randint(50, 2000, len(keywords))
    return pd.DataFrame(data)

def process_crosslinks(df, similarity_threshold=5):
    """Generate crosslinks for New Look products"""
    results = []
    
    for _, target in tqdm(df.iterrows(), total=len(df)):
        target_text = f"{target['Target Title']} {target['Category']} {target['Subcategory']}"
        
        for _, source in df.iterrows():
            if target['Redirect URL'] != source['Redirect URL']:
                source_text = f"{source['Target Title']} {source['Category']} {source['Subcategory']}"
                similarity = get_similarity(target_text, source_text)
                
                if similarity >= similarity_threshold:
                    results.append({
                        'Target URL': target['Redirect URL'],
                        'Target Title': target['Target Title'],
                        'Target Category': target['Category'],
                        'Source URL': source['Redirect URL'],
                        'Source Title': source['Target Title'],
                        'Source Category': source['Category'],
                        'Similarity': similarity,
                        'Target Searches': target['avg_monthly_searches'],
                        'Source Searches': source['avg_monthly_searches']
                    })
    
    return pd.DataFrame(results)
