import xml.etree.ElementTree as ET
import math
import numpy as np
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import nltk
import json


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict[tag]


"""
what I need to add to index:
    DF: nb of docs that contain the term
    N: nb of documents
    TF: number of occurrences in the doc

    tf maps to the ief_scores in vector
    N: number of documents; needs to be figured out
    DF: needs to be created


we need to recompute everything everytime we add a new website
because the logs would change since N would change as well as DF

So what we will do: whenever we process a document, we will save the tf scores
of that vector

tf_scores: {D1: {ECE: 1...}}



"""


def add_to_index(new_doc_id, new_tf_scores, tf_scores={}, previous_df={}):
    index = {}
    df = previous_df.copy()
    tf_scores[new_doc_id] = new_tf_scores
    nb_of_documents = len(tf_scores)

    # build df
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
    return index


"""
{"D01": {'ece': 1, 'john': 2, 'cramer': 2, 'takagi': 1, 'mark': 1, 'lau': 1},
 "D02": {'coe': 1, 'paul': 2, 'cramer': 2, 'joe': 1, 'mark': 1, 'aub': 1},
 "D03": {'apple': 1, 'john': 1, 'samsung': 2, 'paul': 1, 'mark': 3, 'aub': 1}
 }
"""

temp_index = add_to_index("D03", {'apple': 1, 'john': 1, 'samsung': 2, 'paul': 1, 'mark': 3, 'aub': 1},
                          {'D01': {'ece': 1, 'john': 2, 'cramer': 2, 'takagi': 1, 'mark': 1, 'lau': 1},
                           'D02': {'coe': 1, 'paul': 2, 'cramer': 2, 'joe': 1, 'mark': 1, 'aub': 1}},
                          {'ece': 1, 'john': 1, 'cramer': 2, 'takagi': 1, 'mark': 2, 'lau': 1, 'coe': 1, 'paul': 1,
                           'joe': 1, 'aub': 1})


def query(q, index):
    # include xml query

    list_of_words = q.lower().split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    similarities = {}
    max_value = 0
    max_doc = ""

    for word in list_of_words:
        if word not in stop_words:
            word = lemmatizer.lemmatize(word, get_wordnet_pos(word))  # lemmatization

            if word in index:
                for entry in index[word]:
                    if entry[0] not in similarities:
                        similarities[entry[0]] = entry[1]

                        if similarities[entry[0]] > max_value:
                            max_value = similarities[entry[0]]
                            max_doc = entry[0]

                    else:
                        similarities[entry[0]] += entry[1]

                        if similarities[entry[0]] > max_value:
                            max_value = similarities[entry[0]]
                            max_doc = entry[0]

    return (max_doc, max_value)


query("John Cramer LAU instructor", temp_index)


def xml_to_dict(root, dic, node_id="0"):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    dic[node_id] = {"value": root.tag.lower(), "type": "node"}

    pos = 0
    node_id = node_id + "." + str(pos)

    if root.text:
        if root.text.strip():
            for element in root.text.strip().split(" "):  # tokenize
                if element not in stop_words:  # stop word removal
                    new_element = element.lower()  # lowercase
                    new_element = lemmatizer.lemmatize(new_element, get_wordnet_pos(new_element))  # lemmatization
                    node_id = node_id[:-1 * len(str(abs(pos - 1)))] + str(pos)
                    dic[node_id] = {"value": new_element, "type": "content"}
                    pos += 1

    if root.items():
        for att in root.items():
            # In the below statement, we do pos-1 to get the length of the previous position. Because if we are now 10, len would give 2, but in fact, we want to increment 0-0-9, so we need a length of 1. We also add an abs so that if it was 0 the length of the string doesnt give back 2.
            node_id = node_id[:-1 * len(str(abs(pos - 1)))] + str(pos)

            dic[node_id] = {"value": att[0].lower(), "type": "attributeType"}
            dic[node_id + ".0"] = {"value": att[1].lower(), "type": "attributeContent"}
            pos += 1

    if list(root):
        for child in root:
            node_id = node_id[:-1 * len(str(abs(pos - 1)))] + str(pos)
            xml_to_dict(child, dic, node_id)
            pos += 1

    return dic


def get_children_values(dic, node):  # get leaf children
    children = {}

    pos = 0
    index = node + "." + str(pos)

    while index in dic:

        if index + ".0" not in dic:
            children[dic[index]["value"]] = index
        pos += 1
        index = index[:index.rfind(".") + 1] + str(pos)

    return children


def get_path_product(start_node, end_node, weights):
    # weights format: {"0.1": means weight from from 0 to 0.1
    #                  "0.0.1": weight from 0.0 to 0.0.1  }

    start_index = end_node.find(start_node) + len(start_node) - 1
    found = end_node.find(".", start_index + 2)
    product = 1

    while found != -1:
        product *= weights[(end_node[: found])]
        start_index = found - 1
        found = end_node.find(".", start_index + 2)

    return product


def wagner_fischer(stringA, stringB):
    dist = [[0 for e in range(len(stringB))] for j in range(len(stringA))]

    for i in range(1, len(stringA)):
        dist[i][0] = dist[i - 1][0] + 1
    for j in range(1, len(stringB)):
        dist[0][j] = dist[0][j - 1] + 1

    for i in range(len(stringA)):
        for j in range(len(stringB)):
            costs = []
            costs.append(dist[i - 1][j] + 1)
            costs.append(dist[i][j - 1] + 1)
            costs.append(dist[i - 1][j - 1] + int(stringA[i] != stringB[j]))

            dist[i][j] = min(costs)
    return dist[len(stringA) - 1][len(stringB) - 1]


def generate_weights(dic):
    weights = {}
    for e in dic:
        if len(e) > 1:
            current_value = dic[e]["value"]
            parent = e[: e.rfind(".")]
            parent_value = dic[parent]["value"]
            weights[e] = 1 / wagner_fischer(current_value, parent_value)
    return weights


def vectors(dic, weighting_used=2):
    """
    Parameters
    ----------
    dic : dictionary
        The dictionary processed from XML.
    weighting_used : int
        TF (0), IEF (1), or both (2). Default: 2

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    ief_scores = {}
    tf_scores = {}

    values_mapping = {}

    for e in dic:
        if e + ".0" not in dic:  # leaf node
            parent = e[:e.rfind(".")]

            if parent not in tf_scores:
                tf_scores[parent] = {}

                children = get_children_values(dic, parent)

                for child in children.keys():
                    if child not in ief_scores:
                        ief_scores[child] = 1

                    else:
                        ief_scores[child] += 1

                    if child not in tf_scores[parent]:
                        tf_scores[parent][child] = 1
                    else:
                        tf_scores[parent][child] += 1

                    if child not in values_mapping:
                        values_mapping[child] = [children[child]]
                    else:
                        values_mapping[child].append(children[child])

    # base vectors

    if weighting_used == 2:
        math_formula = lambda e, j: tf_scores[e][j] * (
            math.log(len(tf_scores) / ief_scores[j], 10))

    elif weighting_used == 1:
        math_formula = lambda e, j: math.log(len(tf_scores) / ief_scores[j], 10)

    elif weighting_used == 0:
        math_formula = lambda e, j: tf_scores[e][j]

    vectors = {e: {j: math_formula(e, j) for j in tf_scores[e]}
               for e in tf_scores}

    # augment
    augmented = vectors.copy()
    weights = generate_weights(dic)

    for vector in dic:

        if vector + '.0' in dic:  # if not leaf node, proceed. If leaf node, don't
            if vector not in augmented:  # check if not present now; if not present and it is NOT a leaf node, add it.
                augmented[vector] = {}
            for dimension in ief_scores.keys():
                s = 0  # sum

                if dimension not in augmented[
                    vector]:  # if the dimension is not there already; if it is, then the weight would be 1, so might as well not change it.
                    indices = values_mapping[
                        dimension]  # get the indices of where the dimension can be found; this is to get the path as well as the w.
                    for i in indices:
                        product = 1
                        if i[:len(vector)] == vector:  # check if we are a grandparent to that node

                            product *= get_path_product(vector, i, weights)  # get the augmentations

                            # get the weight of the original one
                            parent_node = i[:i.rfind(".")]
                            product *= vectors[parent_node][dimension]

                            s += product

                    augmented[vector][dimension] = s

    return augmented


def uniformize(vector_1, vector_2):
    dimensions_mapping = {}
    mapping_index = 0

    total_set = set(vector_1.keys()).union(set(vector_2.keys()))
    out_1 = np.zeros(len(total_set))
    out_2 = np.zeros(len(total_set))

    for e in total_set:
        if e in vector_1:
            out_1[mapping_index] = vector_1[e]
        else:
            out_1[mapping_index] = 0

        if e in vector_2:
            out_2[mapping_index] = vector_2[e]
        else:
            out_2[mapping_index] = 0

        dimensions_mapping[e] = mapping_index
        mapping_index += 1

    return out_1, out_2, dimensions_mapping


b = ET.parse(r"C:\Users\Anton\desktop\test.txt")
rootb = b.getroot()
c = ET.parse(r"C:\Users\Anton\desktop\test - Copy.txt")
rootc = c.getroot()

datab = {}
xml_to_dict(rootb, datab)
vector_1 = vectors(datab)

datac = {}
xml_to_dict(rootc, datac)
vector_2 = vectors(datac)

print(uniformize(vector_1["0"], vector_2["0"]))