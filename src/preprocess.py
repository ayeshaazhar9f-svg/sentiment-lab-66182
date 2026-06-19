"""Text preprocessing utilities for legal notice classification."""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess(text):
    """Clean and normalize legal notice text.

    Applies HTML tag removal, lowercasing, punctuation removal,
    tokenisation, stopword removal, and lemmatisation in order.

    Args:
        text (str): Raw input string.

    Returns:
        str: Cleaned, lemmatized string.
    """
    text = re.sub(r'<[^>]+>', '', text)          # HTML tag removal: notices may contain markup artefacts
    text = text.lower()                            # Lowercasing: avoid treating 'Contract' and 'contract' as different tokens
    text = re.sub(r'[^a-z\s]', '', text)          # Remove punctuation/numbers: not informative for word-level features
    tokens = text.split()                          # Tokenisation: split into words for further processing
    tokens = [t for t in tokens if t not in stop_words]   # Stopword removal: removes common words with little discriminative value
    tokens = [lemmatizer.lemmatize(t) for t in tokens]    # Lemmatisation: reduces words to dictionary form, reduces vocabulary size
    return ' '.join(tokens)
