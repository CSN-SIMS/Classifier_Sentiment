#Lukas and Jenjira
#Created by following¨and making neacecary changes to a guide at https://pythonprogramming.net
#Work in progress
import os

import nltk
import random
import nltk as nltk
from nltk.corpus import movie_reviews, stopwords
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
import os
from nltk.tokenize import word_tokenize
import re
from nltk.classify import ClassifierI
from statistics import mode

#Ärver från ClassifierI från NLTK, används för att räkna vad alla algoritmer bedömmer och låter varje algoritm 'rösta' på resultatet
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        # Utkommenterat för att returnera neutralt när confidence är lågt.
        # if(mode(votes) == "neg" and votes.count('pos') == 3):
        # return "Neutral"
        # elif(mode(votes) == "pos" and votes.count("neg") == 3):
        # return "Neutral"
        # else:
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

#Filename syntax example /picklefiles/features.pickle
def loadPickleFile(filename):
    file = open(filename, "rb")
    returnObj = pickle.load(file)
    file.close()
    return returnObj

def savePickle(saveObj, filename):
    save_obj = open(filename, "wb")
    pickle.dump(saveObj, save_obj)
    save_obj.close()

#posdirectory syntax example "txt_sentoken/pos/", same syntax for negdirectory. The positive and negative text files cannot be in the same directory.
def loadTrainingDataFromdDirectory(posdirectory, negdirectory):
    all_words = []
    documents = []
    stop_words = list(set(stopwords.words('english')))
    allowed_word_types = ["J"]

    files_pos = os.listdir(posdirectory)
    files_pos = [open(posdirectory + "/" + f, 'r').read() for f in files_pos]
    files_neg = os.listdir(negdirectory)
    files_neg = [open(negdirectory + "/" + f, 'r').read() for f in files_neg]

    for p in files_pos:
        documents.append((p, "pos"))
        cleaned = re.sub(r'[^(a-zA-Z)\s]', '', p)
        tokenized = word_tokenize(cleaned)
        stopped = [w for w in tokenized if not w in stop_words]
        # parts of speech tagging for each word
        pos = nltk.pos_tag(stopped)
        for w in pos:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())

    for p in files_neg:
        documents.append((p, "neg"))
        cleaned = re.sub(r'[^(a-zA-Z)\s]', '', p)
        tokenized = word_tokenize(cleaned)
        stopped = [w for w in tokenized if not w in stop_words]
        # parts of speech tagging for each word
        neg = nltk.pos_tag(stopped)
        for w in neg:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())

    all_words = nltk.FreqDist(all_words)
    word_features = list(all_words.keys())[:5000]

    return word_features

def loadTestingdataFromDirectory(posdir, negdir):
    #Den här delen avnänds för att hämta testsettet
    testdoc = []
    alltestwords = []
    stop_words = list(set(stopwords.words('english')))
    allowed_word_types = ["J"]

    test_pos = os.listdir(posdir)
    test_pos = [open(posdir + '/' + f, 'r').read() for f in test_pos]
    test_neg = os.listdir(negdir)
    test_neg = [open(negdir + '/' + f, 'r').read() for f in test_neg]

    for p in test_pos:
        testdoc.append((p, "pos"))
        cleaned = re.sub(r'[^(a-zA-Z)\s]', '', p)
        tokenized = word_tokenize(cleaned)
        stopped = [w for w in tokenized if not w in stop_words]
        # parts of speech tagging for each word
        pos = nltk.pos_tag(stopped)
        for w in pos:
            alltestwords.append(w[0].lower())

    for p in test_neg:
        testdoc.append((p, "neg"))
        cleaned = re.sub(r'[^(a-zA-Z)\s]', '', p)
        tokenized = word_tokenize(cleaned)
        stopped = [w for w in tokenized if not w in stop_words]
        # parts of speech tagging for each word
        neg = nltk.pos_tag(stopped)
        for w in neg:
            alltestwords.append(w[0].lower())
    alltestwords = nltk.FreqDist(alltestwords)
    test_features = list(alltestwords.keys())

    words = set(testdoc)
    return test_features


def printAndTrainClassifiers(trainingset, testingset, classifier, MNB_classifier, BernoulliNB_classifier, LogisticRegression_classifier, SGDClassifier_classifier, LinearSVC_classifier, NuSVC_classifier):
    print("Basic classifier accuracy percent:",(nltk.classify.accuracy(classifier, testingset))*100)

    MNB_classifier.train(trainingset)
    print("Multinomial Naive Bayes accuracy:", (nltk.classify.accuracy(MNB_classifier, testingset))*100)

    BernoulliNB_classifier.train(trainingset)
    print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testingset))*100)

    LogisticRegression_classifier.train(trainingset)
    print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testingset))*100)

    SGDClassifier_classifier.train(trainingset)
    print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testingset))*100)

    LinearSVC_classifier.train(trainingset)
    print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testingset))*100)

    NuSVC_classifier.train(trainingset)
    print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testingset))*100)

    voted_classifier = VoteClassifier(classifier,
                                      NuSVC_classifier,
                                      BernoulliNB_classifier,
                                      LinearSVC_classifier,
                                      SGDClassifier_classifier,
                                      MNB_classifier,
                                      LogisticRegression_classifier)
    print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testingset)) * 100)

    return voted_classifier

def find_testfeatures(testdoc, test_features):
    words = word_tokenize(testdoc)
    testingfeatures = {}
    for w in test_features:
        testingfeatures[w] = (w in words) #Värt att titta på...
    return testingfeatures

#Print all classifications and their confidence
def printVoteConfidence(voted_classifier, testingset):
    for i in testingset:
        print("Classification:", voted_classifier.classify(i[0]), "Confidence %:",voted_classifier.confidence(i[0])*100)

def find_features(document, wordfeatures):
    words = set(document)
    features = {}
    for w in wordfeatures:
        features[w] = (w in words)

    return features

def loadPremadeMovieReviews():
    all_words = []

    for w in movie_reviews.words():
        all_words.append(w.lower())
    all_words = nltk.FreqDist(all_words)
    word_features = list(all_words.keys())[:3000]
    return word_features
