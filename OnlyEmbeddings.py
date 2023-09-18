import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import json
import pandas as pd
# Scikit-learn importings
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import joblib
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import re, os, string
import requests
from keybert import KeyBERT
import yake
import math

path = "C:/Users/abdul/OneDrive/Documents/New-Sample-Flask/"

df = pd.read_json(f"{path}zaman_with_keywords.json", encoding='utf-8')
print(df["Yake"])

# # Read the contents of the source JSON file
# with open(f"{path}zaman_with_keyword.json", 'r', encoding='utf-8') as file:
#     source_data = json.load(file)

# # Read the contents of the destination JSON file
# with open(f"{path}zaman_with_keywords.json", 'r', encoding='utf-8') as file:
#     destination_data = json.load(file)

# # Extract desired columns from the source data and add to the destination data
# for source_item, destination_item in zip(source_data, destination_data):
#     destination_item['TFIDF'] = source_item['TFIDF']  # Replace 'new_column' with the desired column name
#     destination_item['Keybert'] = source_item['Keybert']  # Replace 'new_column' with the desired column name
#     destination_item['Yake'] = source_item['Yake']  # Replace 'new_column' with the desired column name

# # Convert the modified data structure to JSON format
# new_json = json.dumps(destination_data, indent=4)

# # Write the JSON data to the destination file
# with open(f"{path}zaman_with_keywords.json", 'w', encoding='utf-8') as file:
#     file.write(new_json)