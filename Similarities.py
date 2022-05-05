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
from scipy import spatial
from scipy import stats
from numpy import dot
from numpy.linalg import norm

nlp = spacy.load("en_core_web_md")

def jaccardSimilarity(doc1, doc2):  # Jaccard Similarity = 1 - Jaccard Distance
    vector1 = np.array(doc1)
    vector2 = np.array(doc2)
    return 1 - scipy.spatial.distance.jaccard(vector1, vector2)


def euclideanSimilarity(doc1, doc2):
    vector1 = np.array(doc1)
    vector2 = np.array(doc2)
    return scipy.spatial.distance.euclidean(vector1, vector2)


def manhattanSimilarity(doc1, doc2):
    vector1 = np.array(doc1)
    vector2 = np.array(doc2)
    return scipy.spatial.distance.cityblock(vector1, vector2)


def diceSimilarity(doc1, doc2):  # Dice Similarity = 1 - Dice Distance
    vector1 = np.array(doc1)
    vector2 = np.array(doc2)
    return 1 - scipy.spatial.distance.dice(vector1, vector2)


def cosineSimilarity(doc1, doc2):
    vector1 = np.array(doc1)
    vector2 = np.array(doc2)
    cosine_sim = dot(doc1, doc2) / (norm(vector1) * norm(vector2))
    return cosine_sim


def PCC(doc1, doc2):
    vector1 = np.array(doc1)
    vector2 = np.array(doc2)
    return abs(scipy.stats.pearsonr(vector1, vector2)[0])