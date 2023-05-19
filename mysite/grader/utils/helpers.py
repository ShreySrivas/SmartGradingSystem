import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pandas as pd
import enchant
import language_tool_python  


working_directory = os.getcwd()
stop_words = set(stopwords.words("english"))

def essay_to_wordlist(essay_v, remove_stopwords):
    """Remove the tagged labels and word tokenize the sentence."""
    essay_v = re.sub("[^a-zA-Z]", " ", essay_v)
    words = essay_v.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)

def essay_to_sentences(essay_v, remove_stopwords):
    """Sentence tokenize the essay and call essay_to_wordlist() for word tokenization."""
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(essay_v.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(essay_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

def makeFeatureVec(words, model, num_features):
    """Make Feature Vector from the words list of an Essay."""
    featureVec = np.zeros((num_features,),dtype="float32")
    num_words = 0.
    index_to_key_set = set(model.index_to_key)
    for word in words:
        if word in index_to_key_set:
            num_words += 1
            featureVec = np.add(featureVec,model[word])        
    featureVec = np.divide(featureVec,num_words)
    return featureVec

def getAvgFeatureVecs(essays, model, num_features):
    """Main function to generate the word vectors for word2vec model."""
    counter = 0
    essayFeatureVecs = np.zeros((len(essays),num_features),dtype="float32")
    for essay in essays:
        essayFeatureVecs[counter] = makeFeatureVec(essay, model, num_features)
        counter = counter + 1
    return essayFeatureVecs

def handle_uploaded_file(file):
    file_path = 'media/'+ file.name
    with open(os.path.join(working_directory,file_path), 'r') as new_file:
        keywords = new_file.readlines()
    new_file.close()
    keywords = ''.join(keywords)
    processed_keywords = []
    processed_keywords.append(essay_to_wordlist( keywords, remove_stopwords=True ))
    expanded_keywords = []
    for keyword in processed_keywords[0]:
        keyword_synonyms = []
        synonyms = wordnet.synsets(keyword)
        for syn in synonyms:
            for lemma in syn.lemmas():
                synonym = lemma.name().lower()
                if synonym not in keyword_synonyms:
                    keyword_synonyms.append(synonym)
        expanded_keywords.append(keyword_synonyms)
    return list(expanded_keywords)

def sort_coo(coo_matrix):
    """Sort a dict with highest score"""
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topk):
    """get the feature names and tf-idf score of top n items"""

    score_vals = []
    feature_vals = []
        # word index and corresponding tf-idf score
    for idx, score in sorted_items: 
            #keep track of feature name and its corresponding score
        score_vals.append(score)
        feature_vals.append(feature_names[idx])
        #create a tuples of feature, score
    results= {}

    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    return results

def get_essay_keywords(vectorizer, feature_names, doc, TOP_K_KEYWORDS):
    """Return top k keywords from a doc using TF-IDF method"""
    #generate tf-idf for the given document
    tf_idf_vector = vectorizer.transform(doc)
    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())
    #extract only TOP_K_KEYWORDS
    keywords=extract_topn_from_vector(feature_names,sorted_items,TOP_K_KEYWORDS)
    return list(keywords.keys())

def calculate_score(essay, keywords):
    keyword_size = len(keywords)
    #print(keywords)    
    df = pd.read_csv(os.path.join(working_directory,"media/Processed_data.csv"))
    df.drop("Unnamed: 0",inplace=True,axis=1)

    list_of_essays = df['clean_essay'].tolist()
    vectorizer = TfidfVectorizer(stop_words=stop_words, smooth_idf=True, use_idf=True)
    vectorizer.fit_transform(list_of_essays)
    feature_names = vectorizer.get_feature_names()
    essay_keywords = get_essay_keywords(vectorizer, feature_names, essay[0], keyword_size*2)
    #print(essay_keywords)

    count = 0
    keyword_score = 0
    iter = 0
  
    for essay_word in essay_keywords:
        iter = count
        while iter < keyword_size:
            if essay_word in keywords[iter]:
                #print(essay_word)
                keyword_score += 10/(keyword_size)
                count += 1
                break
            iter = iter + 1

        if count >= keyword_size:
            break
    return keyword_score


nltk.download('stopwords')  

def sent2word(x):
    x=re.sub("[^A-Za-z0-9]"," ",x)
    words=nltk.word_tokenize(x)
    return words

def essay2word(essay):
    essay = essay.strip()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw = tokenizer.tokenize(essay)
    final_words=[]
    for i in raw:
        if(len(i)>0):
            final_words.append(sent2word(i))
    return final_words
        

def no_of_words(essay):
    count=0
    for i in essay_to_wordlist(essay, remove_stopwords=True):
        count=count+len(i)
    return count


def count_spelling_errors(clean_essay):
    clean_essay=clean_essay.lower()
    new_essay = re.sub("[^A-Za-z0-9]"," ",clean_essay)
    new_essay = re.sub("[0-9]","",new_essay)
    count=0
    all_words = new_essay.split()
    dictionary = enchant.Dict("en_US")
    for word in all_words:
        if not dictionary.check(word):
            count+=1
    return count

def count_grammatical_errors(clean_essay):
  
    my_tool = language_tool_python.LanguageTool('en-US')  
    count = 0
    for line in clean_essay:
        matches = my_tool.check(line)
        count += len(matches)    
    return count
        

def count_redundancies(clean_essay):
    redundancies = 0
    wordlist = essay_to_wordlist(clean_essay, remove_stopwords=True)
    for i in range(0, len(wordlist)):  
        count = 1
        for j in range(i+1, len(wordlist)):  
            if(wordlist[i] == (wordlist[j])):  
                count = count + 1
                wordlist[j] = "0"
                
        if(count > 1 and wordlist[i] != "0"):  
            redundancies += 1
    return redundancies
