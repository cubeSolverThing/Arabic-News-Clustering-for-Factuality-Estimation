import jsonlines
import json
import matplotlib.pyplot as plt
from scipy import spatial
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, session, jsonify
import faiss  
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import re, os, string
import pandas as pd
# Scikit-learn importings
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import joblib
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import requests
from keybert import KeyBERT
import yake
from wordcloud import WordCloud
from bidi.algorithm import get_display
import arabic_reshaper
from PIL import Image, ImageOps
import ktrain
from ktrain import text
import tensorflow
from emoji import UNICODE_EMOJI
from sklearn.metrics import pairwise_distances


# from flask_bootstrap import Bootstrap 
# from flask_ngrok import run_with_ngrok

path = "C:/Users/abdul/OneDrive/Documents/New-Sample-Flask/"
df = pd.read_json(f"{path}zaman_with_keyword.json", encoding='utf-8')
clf = ktrain.load_predictor(f'{path}Predictor')

drop_menu_info_path = f'{path}Dropdown_menu_info.txt'
drop_menu_info = open(drop_menu_info_path, 'r', encoding='utf-8').readlines()
    
df_rankings = pd.read_csv(f"{path}seq-ranked-domains.csv")
finalGroup = []
finalGroup.append([1, 'bbc.co.uk'])
#Constants
PUNCTUATION = """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""" 
TOP_K_KEYWORDS = 10 # top k number of keywords to retrieve in a ranked document
STOPWORD_PATH = "/kaggle/input/stopwords/stopwords.txt"
PAPERS_PATH = "/kaggle/input/nips-papers-1987-2019-updated/papers.csv"

file_path = f"{path}cluster_dist_dict.json"
# Read the JSON data from file into a dictionary
with open(file_path, "r",  encoding='utf-8') as json_file:
    cluster_dist_dict = json.load(json_file)
# cluster_df = pd.read_json(f"{path}Cluster_df.json", encoding='utf-8')
most_representative_sent = []
for i in range(len(cluster_dist_dict)):
  most_representative_sent.append(cluster_dist_dict[str(i)][0][0])



for i in range(df_rankings.shape[0]):
   group=[df_rankings.iloc[i,0],df_rankings.iloc[i,1]]
   finalGroup.append(group)  
df_rankings = pd.DataFrame(finalGroup, columns=["Rank","Domain"])

sourceRankDict = {}
for j in range (df_rankings.shape[0]):
    sourceRankDict[df_rankings.iloc[j,1]] = df_rankings.iloc[j,0]

def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()
    return cleaned_text

df['content'] = df['content'].apply(remove_html_tags)
df['publishedAt'] = pd.to_datetime(df['publishedAt'])
df.fillna("N/A", inplace=True)

model = SentenceTransformer(f"{path}bert-large-arabertv2")
sentence_embeddings = np.load(f"{path}sentence_embeddingsUpd.npy")

def search2(word, embeddings, score, numArticles):
  word = word.strip()
  if (word == "" or word.isspace()):
    return "Type a Query Please"
  else:
    k = int(numArticles)
    d = 1024  
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    relatedArt = []
    encoded =  model.encode([word])
    D, I = index.search(encoded, k) 
    if (score == "" or score.isspace()):
            for i in list(I[0]):
                
                sourceDom = df.iloc[i,15]
                country = df.iloc[i,14]
                simScore = round(1 - spatial.distance.cosine(list(encoded[0]),list(model.encode(df.iloc[i,4]))),3)

                if (country == "" or country.isspace()):
                    country = "N/A"
                if (df.iloc[i,9] == "" or df.iloc[i,9].isspace()):
                    df.iloc[i,9] = "N/A"
                if (sourceDom in sourceRankDict):
                    domainRank =  int(np.where(df_rankings[["Domain"]] == df.iloc[i,15])[0][0])                   
                    artDeets = [sourceRankDict[df.iloc[i,15]],df.iloc[i,16]["name"],df.iloc[i,18],df.iloc[i,4],df.iloc[i,9][:150],country,df.iloc[i,2], simScore, df.iloc[i,11], "https://" + df.iloc[i,15], df.iloc[i,0]]
                    relatedArt.append(artDeets)  
                else:
                    domainRank =  "N/A"
                    artDeets = [domainRank,df.iloc[i,16]["name"],df.iloc[i,18],df.iloc[i,4],df.iloc[i,9][:150],country,df.iloc[i,2], simScore, df.iloc[i,11], "https://" + df.iloc[i,15], df.iloc[i,0]]
                    relatedArt.append(artDeets)         

    elif (0 < float(score) < 1):
    
            for i in list(I[0]):
                
                sourceDom = df.iloc[i,15]
                simScore = round(1 - spatial.distance.cosine(list(encoded[0]),list(model.encode(df.iloc[i,4]))),3)
                country = df.iloc[i,14]

                if (country == "" or country.isspace()):
                    country = "N/A"

                if (df.iloc[i,9] == "" or df.iloc[i,9].isspace()):
                    df.iloc[i,9] = "N/A"
                if (sourceDom in sourceRankDict):   
                    if (simScore >= float(score)):
                        domainRank =  int(np.where(df_rankings[["Domain"]] == df.iloc[i,15])[0][0])
                        artDeets = [sourceRankDict[df.iloc[i,15]], df.iloc[i,16]["name"],df.iloc[i,18],df.iloc[i,4],df.iloc[i,9][:150],country,df.iloc[i,2], simScore, df.iloc[i,11], "https://" + df.iloc[i,15], df.iloc[i,0]]
                        relatedArt.append(artDeets)  
                else:
                    if (simScore >= float(score)):
                            domainRank =  "N/A"
                            artDeets = [domainRank, df.iloc[i,16]["name"],df.iloc[i,18],df.iloc[i,4],df.iloc[i,9][:150],country,df.iloc[i,2], simScore, df.iloc[i,11], "https://" + df.iloc[i,15], df.iloc[i,0]]
                            relatedArt.append(artDeets)  

    else:
      return "The Similarity Score has to be between 0 and 1"
    
    if (relatedArt == [] and 0 < float(score) < 1):
      return "No Articles Match Your Query"
    else:
        relatedArt = sorted(relatedArt, key=lambda x: x[7], reverse=True)
        relatedDF = pd.DataFrame(relatedArt, columns=['Rank','SourceName', 'Source Image','Title','Summary','Country','DatePublished','Similarity Score','URL','Domain', 'ID']) 
        return relatedDF.values.tolist()

def compareSim(sentence, sentence_two):
   
   if (sentence == "" or sentence.isspace() or sentence_two == "" or sentence_two.isspace()):
      return "Insert a Complete Input Please"
   else:
    encode =  model.encode([sentence])
    encode_sen_two =  model.encode([sentence_two])
    simScore = round(1 - spatial.distance.cosine(list(encode[0]),list(encode_sen_two[0])),3)
    return "Similarity Score: " + str(simScore)

   
def get_stopwords_list(stop_file_path):
    """load stop words """
    
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return list(frozenset(stop_set))

def clean_text(text):
    """Doc cleaning"""
    
    # Lowering text
    text = text.lower()
    
    # Removing punctuation
    text = "".join([c for c in text if c not in PUNCTUATION])
    
    # Removing whitespace and newlines
    text = re.sub('\s+',' ',text)
    
    return text


def sort_coo(coo_matrix):
    """Sort a dict with highest score"""
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature, score
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results
def get_keywords(vectorizer, feature_names, doc):
    """Return top k keywords from a doc using TF-IDF method"""

    #generate tf-idf for the given document
    tf_idf_vector = vectorizer.transform([doc])
    
    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())

    #extract only TOP_K_KEYWORDS
    keywords=extract_topn_from_vector(feature_names,sorted_items,TOP_K_KEYWORDS)
    
    return list(keywords.keys())

def articleSpecAnalysis(idNum):
    result = []
    dff = {}
    content = df.iloc[idNum,9]
    TFIDFWords = df.loc[idNum,"TFIDF"]
    keybert_keywords = df.loc[idNum,"Keybert"]
    yake_keywords = df.loc[idNum,"Yake"]
    allWords = (TFIDFWords + "/" + keybert_keywords + "/" + yake_keywords).split("/")
    TFIDFWords = TFIDFWords.split("/")
    tuple_list = [eval(tuple_string) for tuple_string in TFIDFWords]
    TFIDFWords = tuple_list
    strings_only = [item[0] for item in TFIDFWords] 
    TFIDFWords = strings_only[::-1]
    keybert_keywords = keybert_keywords.split("/")
    tuple_list = [eval(tuple_string) for tuple_string in keybert_keywords]
    keybert_keywords = tuple_list
    strings_only = [item[0] for item in keybert_keywords]
    keybert_keywords = strings_only[::-1]
    yake_keywords = yake_keywords.split("/")
    tuple_list = [eval(tuple_string) for tuple_string in yake_keywords]
    yake_keywords = tuple_list
    strings_only = [item[0] for item in yake_keywords]
    yake_keywords = strings_only
    # dff['full_text'] = content
    dff['top_keywords_tf'] = 'tf'+', '.join(TFIDFWords)
    dff['top_keywords_yake'] = 'ya'+', '.join(yake_keywords)
    dff['top_keywords_keybert'] = 'ke'+', '.join(keybert_keywords)
    result.append(dff)
    final = pd.DataFrame(result)
    return [final.values.tolist(), allWords]



def remove_html_tags(text):
      soup = BeautifulSoup(text, "html.parser")
      cleaned_text = soup.get_text()
      return cleaned_text

def remove_links(text):
    pattern = r'(?:https?|ftp)://[\n\S]+'
    return re.sub(pattern, '', text)

def normalize_tweet(text):
    edited_text = text.replace('أ', 'ا').replace('ى', 'ي').replace('ة', 'ه').replace("إ", "ا")
    edited_text = edited_text.replace('\u0640', '')
    return re.sub('[\u064B-\u065F]', '', edited_text)

def remove_numbers(text):
    pattern = r'\d+'
    return re.sub(pattern, '', text)

def remove_mentions(text):
    pattern = r'@[\w_]+'
    return re.sub(pattern, '', text)

def remove_punctuation(text):
    pattern = r'[!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]'
    return re.sub(pattern, '', text)

def remove_non_arabic(text):
# Match all characters that are not Arabic letters
  pattern = "abcdefghijklmnopqrstuvwxyz"
  result = ""
  for char in str(text):
    if char.lower() not in pattern:
      result += char
  return result

def remove_emojis(text):
  result = ""
  for char in text:
    if char not in UNICODE_EMOJI['en'].keys():
      result += char
  return result

def remove_hashtags(text):
    pattern = re.sub(r'#\w+', '', text)
    return pattern

def apply_normalization_and_link_removal(tweet):
  output_text = tweet
  output_text = normalize_tweet(output_text)
  output_text = remove_links(output_text)
  output_text = remove_numbers(output_text)
  output_text = remove_mentions(output_text)
  output_text = remove_hashtags(output_text)
  output_text = remove_punctuation(output_text)
  output_text = remove_emojis(output_text)
  return output_text

def remove_questionmarks(text):
  text = text.replace("? ? ? ?", "")
  return text

def disconnect_ar_eng(text):
    pattern = r'([^\u0600-\u06FF\u0750-\u077F\s]+|[^\sa-zA-Z]+)'
    result = re.sub(pattern, r' \1 ', text)
    return result.strip()

def clean_text(text):
  text = apply_normalization_and_link_removal(text)
  text = re.sub(r"@|#|/|•|;|:|—|–|\(|\)|\[|\]|\{|\}|\||\+|\*|\-|\'|‘|“|”|‘|’|\.\.\.|\xa0|\t|\-|\_|\:|\…|»|«|\n|،|●|🇵🇸|🇪🇬", " ", text)
  text = re.sub(r"\n", " ", text)
  text = disconnect_ar_eng(text)
  text = re.sub(r'\s+', ' ', text)
  return text

api_key1 = "WIomaoMipJvsSUrqdw" # Test key

def farasa_functions(f, text, key):
    url = f'https://farasa.qcri.org/webapi/{f}/'
    payload = {'text': text, 'api_key': key}
    data = requests.post(url, data=payload)
    result = json.loads(data.text)
    result = " ".join(result["text"])
    return result

def text_processing(text):
  text = remove_html_tags(text)
  text = clean_text(text)
  try:
    text = farasa_functions("lemmatization", text, api_key1)
  except:
    text = text
  text = remove_questionmarks(text)
  return text



    

    
    
app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = 'shaqboom6587t'
dfIndexNum = None

@app.route('/')
def QuerySite():
    session['yesPercent'] = 0
    return render_template("QuerySite.html")

@app.route('/submit', methods=['POST'])
def submit():
    similarity = request.form['similarity']
    query = request.form['query']
    articles = request.form['articles']
    query_search = search2(query,sentence_embeddings,similarity,articles)
    query = text_processing(query)
    prediction = clf.predict(query, return_proba=True)
    # Convert the rounded array to a Python list
    prediction = prediction.tolist()
    yesWhole = round(prediction[1] * 19)
    session['yesPercent'] = yesWhole
    if query_search != "The Similarity Score has to be between 0 and 1" and query_search != "Type a Query Please":
        return render_template("QuerySite.html",query=query,similarity=similarity, articles=articles, query_search = query_search, yesPercent = yesWhole, yesPercentage=str(round(prediction[1]*100, 1))+"%" )
    elif (query_search == "The Similarity Score has to be between 0 and 1"):
        session['yesPercent'] = 0
        return render_template("QuerySite.html", query_search = "The Similarity Score has to be between 0 and 1" )
    else:
        session['yesPercent'] = 0
        return render_template("QuerySite.html", query_search = "Type a Query Please" )

@app.route('/yes', methods=['POST'])
def danielBryan():
    yesWhole = session.get('yesPercent')
    return jsonify(yesPercent=yesWhole)

@app.route('/sim')
def simSite():
    return render_template("Similarity Test.html")

@app.route('/simFunc', methods=['POST'])
def simComp():
    sentence = request.form['first']
    sentence_two = request.form['second']
    score = compareSim(sentence,sentence_two)
    if score != "Insert a Complete Input Please":
        return render_template("Similarity Test.html",sentence=sentence,sentence_two=sentence_two, score = score)
    else:
        return render_template("Similarity Test.html",sentence=sentence,sentence_two=sentence_two, score = "Insert a Complete Input Please")

@app.route('/simTestRando')
def randoSimScore(): 
    randomNum = np.random.randint(0,df.shape[0])
    randomNum1 = np.random.randint(0,df.shape[0])

    if (randomNum == randomNum1):
        randomNum = np.random.randint(0,df.shape[0])
        randomNum1 = np.random.randint(0,df.shape[0])

    sentence = df.iloc[randomNum,4]
    sentence_two = df.iloc[randomNum1,4]
        
    return render_template('Similarity Test.html',sentence = sentence, sentence_two = sentence_two)
   
@app.route('/random') 
def randomSample():
    session['yesPercent'] = 0
    randomNum = np.random.randint(0,df.shape[0])
    query = df.iloc[randomNum,4]
    return render_template('QuerySite.html', query = query)
      
@app.route('/stats')
def stats():
    return render_template('StatisticsWebPage.html')

@app.route('/header')
def header():
    return render_template('Header.html')

@app.route('/analysis')
def openAnalys():
    PossibleNER = {}
    important = getContent()
    imageImp = important[0]
    title = important[1]
    content = important[2]
    NERTing = farasa_functions('ner',content, api_key1).split("/O")
    

    for j in NERTing:
        if "/" in j:
            splitting = j.split("/")
            splitAgain = splitting[1].split(" ")
            if (splitAgain[0] not in PossibleNER):
                PossibleNER[splitAgain[0]] = splitting[0]
            else:
                tempList = []
                if isinstance(PossibleNER[splitAgain[0]], list):
                    tempList = PossibleNER[splitAgain[0]]
                    tempList.append(splitting[0])
                    PossibleNER[splitAgain[0]] = tempList
                else:
                    tempList = [PossibleNER[splitAgain[0]], splitting[0]]
                    PossibleNER[splitAgain[0]] = tempList
    print(PossibleNER)
    for key, value in PossibleNER.items():
        # Convert the list to a set to remove duplicates and then back to a list
        if (isinstance(value,list)):
            unique_values = list(set(value))
            # Update the dictionary with the modified list
            PossibleNER[key] = unique_values
    print(PossibleNER)
    
    # print(NERTing[0], "-" + NERTing[1])
    finale = important[3]
    words = important[4]
    tuple_list = [eval(tuple_string) for tuple_string in words]

    # Sample data as a list of tuples (value, word)
    
    # Preprocess the data to extract individual words and apply right-to-left formatting
    preprocessed_data = []
    for word, value in tuple_list:
        reshaped_word = arabic_reshaper.reshape(word)  # Reshape the Arabic word
        bidi_word = get_display(reshaped_word)  # Apply right-to-left formatting
        preprocessed_data.append((bidi_word, value))

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, colormap='Dark2', font_path=f"{path}Arab.ttf", background_color='#F3FBF6')
    # Generate word frequencies dictionary from the preprocessed data
    word_frequencies = {word: value for word, value in preprocessed_data}
    wordcloud.generate_from_frequencies(word_frequencies)
    image_path = 'static/wordcloud.png'
    wordcloud.to_file(image_path)
    image = Image.open(image_path)
    border_width = 10
    border_color = '#046307'
    image_with_border = ImageOps.expand(image, border=border_width, fill=border_color)
    image_with_border.save(image_path)
    # Display the word cloud using matplotlib

   
    return render_template('ArticleAnalysis.html', imageImp=imageImp, title=title, content = content, wordcloud=image_path, finale = finale, PossibleNER = PossibleNER)



def getContent():
    url = request.url
    id = url.split("row=")[1]
    dfIndex = df[df['_id'] == id].index
    dfRow = dfIndex[0]
    content = df.iloc[dfRow]["spellchecked_only"]
    title = df.iloc[dfRow]["title"]
    image = df.iloc[dfRow]["urlToImage"]

    tempList = articleSpecAnalysis(dfRow)
    return([image,title,content,tempList[0],tempList[1]])




@app.route('/cluster')
def cluster():
    cluster_names = drop_menu_info
    return render_template("ClusterSite.html", cluster_names=cluster_names)

@app.route('/process_cluster', methods=['POST'])

def process_cluster():
  selected_position = request.json.get('position')
  selected_value = request.json.get('value')
  cluser_list_text = str(selected_value).split("||")
  cluster_list_title = cluser_list_text[2]
  column_name = ""
  for key, values in cluster_dist_dict.items():
    temp = values[0][0]
    if temp.strip() == cluster_list_title.strip():
      column_name = key
      print("Found", key)
      break
  else:
      print("No match found.")
    
  cluster_information = []

  for sent in cluster_dist_dict[column_name]: 
    text = sent
    title = text[0]
    id = text[1]
    center_distance = text[2]
    
    temp_df = df[df['_id'] == id]
    country = temp_df.iloc[0]["country"]
    content = re.sub(r'\s+', ' ', temp_df.iloc[0]["content"])
    sourceDom = temp_df.iloc[0]["domain"]
    source_name = temp_df.iloc[0]["source_meta"]["name"]
    image_url = temp_df.iloc[0]["urlToImage"]
    domainRank = "N/A"
    date = temp_df.iloc[0]["publishedAt"]
    site_url = temp_df.iloc[0]["url"]
    if (content == "" or content.isspace()):
        content = "N/A"
    if (sourceDom in sourceRankDict):
        domainRank =  int(np.where(df_rankings[["Domain"]] == sourceDom)[0][0])                 

    artDeets = [domainRank, source_name, image_url, title, content[:150], country, date, site_url, "https://" + sourceDom, id, str(selected_position), center_distance]
    cluster_information.append(artDeets) 
    cluster_information = sorted(cluster_information, key=lambda x: int(x[0]) if x[0] != 'N/A' else 500,)
  print(len(cluster_information))
  return jsonify(cluster_information)


if __name__ == '__main__':
    app.run(debug=True)