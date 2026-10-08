import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import json

print("Loading data...")
df = pd.read_csv('data/netflix_cleaned.csv')
for col in ['description','director','listed_in','cast']:
    df[col] = df[col].fillna('')

df['combined_features'] = df['listed_in']+' '+df['description']+' '+df['director']+' '+df['cast']

print("Vectorizing...")
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

print("Saving tfidf_matrix.pkl...")
with open('tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)

print("Done! Reduced size drastically.")
