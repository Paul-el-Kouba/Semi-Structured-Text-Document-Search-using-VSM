import scipy
import numpy as np
from numpy import dot
from numpy.linalg import norm
import scipy.stats as sci


def jaccardDistance(doc1, doc2):  # Jaccard Similarity = 1 - Jaccard Distance
    vector1 = np.array(doc1)
    vector2 = np.array(doc2)
    return scipy.spatial.distance.jaccard(vector1, vector2)


def euclideanDistance(doc1, doc2):
    vector1 = np.array(doc1)
    vector2 = np.array(doc2)
    return scipy.spatial.distance.euclidean(vector1, vector2)


def manhattanDistance(doc1, doc2):
    vector1 = np.array(doc1)
    vector2 = np.array(doc2)
    return scipy.spatial.distance.cityblock(vector1, vector2)


def diceDistance(doc1, doc2):  # Dice Similarity = 1 - Dice Distance
    vector1 = np.array(doc1)
    vector2 = np.array(doc2)
    return scipy.spatial.distance.dice(vector1, vector2)


def cosineSimilarity(doc1, doc2):
    vector1 = np.array(doc1)
    vector2 = np.array(doc2)
    cosine_sim = dot(doc1, doc2) / (norm(vector1) * norm(vector2))
    return cosine_sim


def PCC(doc1, doc2):
    vector1 = np.array(doc1)
    vector2 = np.array(doc2)
    return abs(sci.pearsonr(vector1, vector2))
