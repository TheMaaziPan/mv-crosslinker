import re
import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_string_and_remove_stopwords(text):
    """Clean text by removing special chars, stopwords, and lemmatizing"""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        words = [lemmatizer.lemmatize(word) for word in text.split() 
                if word not in stop_words and len(word) > 2]
        return ' '.join(words)
    except Exception as e:
        logger.error(f"Text cleaning failed: {e}")
        return text.lower()

def get_similarity(title1, title2):
    """Calculate similarity between two titles (0-10 scale)"""
    if not title1 or not title2:
        return 0.0
    try:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([title1, title2])
        return round(cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 10, 2)
    except Exception as e:
        logger.error(f"Similarity calculation failed: {e}")
        return 0.0

def generate_mock_volumes(keywords, months=None):
    """Generate consistent mock volume data"""
    if not months:
        months = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE',
                'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']
    
    np.random.seed(42)  # For consistent results
    data = {
        'keyword': keywords,
        'avg_monthly_searches': np.random.randint(100, 5000, len(keywords))
    }
    for month in months:
        data[month] = np.random.randint(50, 2000, len(keywords))
    
    return pd.DataFrame(data)

def process_crosslinks(df, similarity_threshold=5):
    """Generate crosslinks ensuring equal array lengths"""
    results = []
    for i, target_row in tqdm(df.iterrows(), total=len(df)):
        target_text = f"{target_row['Title']} {target_row['Category']}"
        
        for j, source_row in df.iterrows():
            if i != j:
                source_text = f"{source_row['Title']} {source_row['Category']}"
                similarity = get_similarity(target_text, source_text)
                
                if similarity > similarity_threshold:
                    results.append({
                        'Target URL': target_row['Redirect URL'],
                        'Source URL': source_row['Redirect URL'],
                        'Target Title': target_row['Title'],
                        'Source Title': source_row['Title'],
                        'Similarity': similarity
                    })
    
    return pd.DataFrame(results)
