import xml.etree.ElementTree as ET
import math
import numpy as np
import json
import spacy
from collections import Counter
import os
import scipy
import time
import scipy as sci
from numpy import dot
from numpy.linalg import norm
from Processing import xml_to_dict
from Similarities import cosineSimilarity
from Processing import uniformize

nlp = spacy.load("en_core_web_md")

# for querying we are treating the xml as text since we didn't incorporate the dewey numbers into the thing

def load_index():
    index_file = open(r"./index.txt", "r")
    index = json.load(index_file)
    index_file.close()
    
    return index

def save_tf_df_index_pos(tf_scores, df_scores, index, pos):
    tf_file = open(r"./tf_scores.txt", "w")
    df_file = open(r"./df.txt", "w")
    index_file = open(r"./index.txt", "w")
    pos_file = open(r"./pos.txt", "w")
    
    
    json.dump(tf_scores, tf_file, indent=4)
    json.dump(df_scores, df_file, indent=4)
    json.dump(index, index_file, indent=4)
    pos_file.write(pos)
    
    tf_file.close()
    df_file.close()
    index_file.close()
    pos_file.close()


def get_new_pos():
    pos_file = open(r"./pos.txt", "r")
    pos = pos_file.read().strip()
    return pos[0] + str(int(pos[1:]) + 1)


def load_tf_df():
    tf_file = open(r"./tf_scores.txt", "r")
    df_file = open(r"./df.txt", "r")

    tf_scores = json.load(tf_file)
    df_scores = json.load(df_file)
    
    tf_file.close()
    df_file.close()
    
    return tf_scores, df_scores
 
def add_to_index(new_tf_scores, tf_scores={}, previous_df={}):
    index = {}
    
    
    new_doc_id = get_new_pos()

        
    df = previous_df.copy()
    tf_scores[new_doc_id] = new_tf_scores
    nb_of_documents = len(tf_scores)
    
    #build df
    for word in new_tf_scores:
        if word not in df:
            df[word] = 1
        else:
            df[word] += 1
        
    for doc_id in tf_scores:
        for element in tf_scores[doc_id]:
            value = tf_scores[doc_id][element] * (
                math.log(nb_of_documents / df[element]))
            
            if element not in index:
                index[element] = [(doc_id, value)]
            else:
                index[element].append((doc_id, value))
                
    save_tf_df_index_pos(tf_scores, df, index, new_doc_id)
    
    return index, tf_scores, df
    
def get_counts(dic):                    #to be used when flattening xml query and when adding to index
    counts = {}
    for e in dic:
        if e + ".0" not in dic:
            value = dic[e]["value"]
            if value:
                if value not in counts:   #if value cz empty string for attribute value might be parsed by xml_to_dic
                    counts[value] = 1
                else:
                    counts[value] += 1
    return counts

def process_collection():
    d = r".\test_docs"
    files = os.listdir(d)
    tf_scores = {}
    df_scores = {}
    count = 0
    for f in files:
        f_dir = os.path.join(d, f)
        b = ET.parse(f_dir)
        rootb = b.getroot()
        data = {}
        root_tf_scores = get_counts(xml_to_dict(rootb, data))
        tf_scores, df_scores = add_to_index(root_tf_scores, tf_scores, df_scores)[1:]     
        
        count += 1
        print(count/len(files) * 100, "% done")


def add_interface(filename):
    b = ET.parse(r"./" + filename)
    rootb = b.getroot()
    datab = {}
    xml_to_dict(rootb, datab)
    
    tf = get_counts(datab)
    add_to_index(tf)

def query(q, q_type, use_index):
    #0: text, 1: xml
    
    #xml query was flattened to be like text
    if q_type == 1:
        temp_dic = {}
        xml = ET.ElementTree(ET.fromstring(q))
        list_of_words = get_counts(xml_to_dict(xml.getroot(), temp_dic))
        
    else:
        list_of_words = dict(Counter(q.lower().split()))
        
    stop_words = nlp.Defaults.stop_words

    
    similarities = {}
    computed_sims = {}
    
    
    if use_index:
        index = load_index()
    else:
        process_collection()
        index = load_index()
        
        
    for word in list_of_words:
        if word not in stop_words:
            tokens = nlp(word)
            for token in tokens:
                word_lemm = token.lemma_
            
                if word_lemm in index:
                    for entry in index[word_lemm]:
                        if entry[0] not in similarities:
                            similarities[entry[0]] = {word_lemm: entry[1]}
                        else:
                            similarities[entry[0]][word_lemm] = entry[1]
                   
    
    for document in similarities:
        vector_1, vector_2, mapping = uniformize(list_of_words, similarities[document])
        
        
        if not np.all((vector_2 == 0)):
            sim = cosineSimilarity(vector_1, vector_2)
            #sim = manhattanDistance(vector_1, vector_2)
            computed_sims[document] = sim
        else:
            computed_sims[document] = 0


    sorted_values = sorted(computed_sims.values())
    keys =  sorted(computed_sims, key=computed_sims.get)
    final = sorted(computed_sims.items(), key=lambda x:x[1], reverse=True)
    return final[:5]