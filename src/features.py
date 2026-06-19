"""Feature extraction utilities: Bag-of-Words and TF-IDF representations."""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_bow_features(X_train, X_test, max_features=5000):
    """Build Bag-of-Words features using CountVectorizer.

    Args:
        X_train (pd.Series): Training text data.
        X_test (pd.Series): Test text data.
        max_features (int): Maximum vocabulary size.

    Returns:
        tuple: (X_train_bow, X_test_bow, fitted_vectorizer)
    """
    vectorizer = CountVectorizer(max_features=max_features)
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)
    return X_train_bow, X_test_bow, vectorizer


def build_tfidf_features(X_train, X_test, max_features=5000):
    """Build TF-IDF features using TfidfVectorizer.

    Args:
        X_train (pd.Series): Training text data.
        X_test (pd.Series): Test text data.
        max_features (int): Maximum vocabulary size.

    Returns:
        tuple: (X_train_tfidf, X_test_tfidf, fitted_vectorizer)
    """
    vectorizer = TfidfVectorizer(max_features=max_features, sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer
