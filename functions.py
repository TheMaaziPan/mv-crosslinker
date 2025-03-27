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

# Initialize NLTK with error handling
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.info("Downloading NLTK data...")
    nltk.download('stopwords')
    nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize NLP components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text Processing Functions
def clean_string_and_remove_stopwords(text):
    """Clean and normalize text by removing stopwords and lemmatizing"""
    if not isinstance(text, str) or not text.strip():
        return None
        
    try:
        # Remove special chars and lowercase
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        # Tokenize, remove stopwords, and lemmatize
        words = [lemmatizer.lemmatize(word) for word in text.split() 
                if word not in stop_words and len(word) > 2]
        return ' '.join(words)
    except Exception as e:
        logger.error(f"Text cleaning failed: {e}")
        return text.lower()  # Fallback to simple lowercase

def get_similarity(title, description):
    """Calculate cosine similarity between texts (0-10 scale)"""
    if not title or not description:
        return 0.0
        
    try:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([title, description])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 10
        return round(similarity, 2)
    except Exception as e:
        logger.error(f"Similarity calculation failed: {e}")
        return 0.0

# Data Processing Functions
def get_ad_volumes(df):
    """Generate mock search volume data"""
    keywords = df['keyword'].dropna().unique()
    months = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE',
             'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']
    
    data = {
        'keyword': np.repeat(keywords, len(months)),
        **{month: np.random.randint(100, 5000, len(keywords)) for month in months}
    }
    
    volume_data = pd.DataFrame(data)
    volume_data['avg_monthly_searches'] = volume_data[months].mean(axis=1)
    return volume_data

def get_gsc_positions():
    """Generate mock GSC data"""
    return pd.DataFrame({
        'keyword': ['dress', 'shoes', 'jacket', 'jeans', 't-shirt'],
        'position': [2.5, 4.1, 7.8, 3.2, 5.4]
    })

# Database Functions
def create_engine():
    """Create SQLAlchemy engine with error handling"""
    try:
        return create_engine('sqlite:///:memory:')  # Default to SQLite
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None
