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
from googletrans import Translator

#Ärver från ClassifierI från NLTK, används för att räkna vad alla algoritmer bedömmer och låter varje algoritm 'rösta' på resultatet, 7 algoritmer från scikit används i implementeringen.
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

        #Ifall vi bedömmer att algoritmen för ofta väljer neg över pos.
        #if(mode(votes) == "neg" and votes.count('neg') == 4):
            #return 'pos'
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

#Inte en använd funktion, finns kvar ifall delar av den blir aktuella.
def loadTrainingDataFromdDirectory(posdirectory, negdirectory):
    all_words = []
    documents = []
    stop_words = list(set(stopwords.words('english')))
    allowed_word_types = ["J"] #Kan inte användas efter att datasettet blir svenskt.
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

#Inte en använd funktion, är kvar ifall delar av den skulle bli aktuella.
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


#Print all classifications and their confidence
def printVoteConfidence(voted_classifier, testingset):
    lidl = 0
    for i in testingset:
        print("Classification:", voted_classifier.classify(i[0]), "Confidence %:",voted_classifier.confidence(i[0])*100)
        lidl+=1
        if(lidl > 39):
            return

def loadPremadeMovieReviews():
    all_words = []

    for w in movie_reviews.words():
        all_words.append(w.lower())
    all_words = nltk.FreqDist(all_words)
    word_features = list(all_words.keys())[:3000]
    return word_features

def find_features(document, wordfeatures):
    words = word_tokenize(document)
    features = {}
    for w in wordfeatures:
        features[w] = (w in words)
    return features


def loadDatasetFromSingleFiles(posfile, negfile):
    short_pos = open(posfile, "r", encoding = "ISO-8859-1").read()
    short_neg = open(negfile, "r", encoding = "ISO-8859-1").read()

    documents = []
    all_words = []

    #J = Adjectives, this has to be changed for when the dataset is in swedish.
    allowed_word_types = ["J"]


    for r in short_pos.split('\n'):
        documents.append((r, "pos"))
        words = word_tokenize(r)
        pos = nltk.pos_tag(words)
        for w in pos:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())

    for r in short_neg.split('\n'):
        documents.append((r, "neg"))
        words = word_tokenize(r)
        pos = nltk.pos_tag(words)
        for w in pos:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())


    #short_pos_words = word_tokenize(short_pos)
    #short_neg_words = word_tokenize(short_neg)

    all_words = nltk.FreqDist(all_words)
    word_features = list(all_words.keys())[:5000]
    savePickle(word_features, "word_features.pickle")
    savePickle(documents, "documents.pickle")

    featuresets = [(find_features(rev, word_features), category) for (rev,category) in documents]
    random.shuffle(featuresets)
    return featuresets

def sentiment(text, voted_classifier, wordfeatures):
    feats = find_features(text, wordfeatures)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)

#Ändra efter språkändring, fixa så att bara txt filer tas in.
def loadTextfilesForAnalysis(directory):
    messageso = []
    fileNames = []
    stop_words = list(set(stopwords.words('english')))
    #allowed_word_types = ["J"]

    files = os.listdir(directory)
    for file in files:
        fileNames.append(file)

    files = [open(directory + "/" + f, 'r').read() for f in files]

    for p in files:
        p.replace('\n', '')#Fungerar inte just nu
        print(p)
        print("______")
        messageso.append(p)

    finalResult = list(zip(fileNames, messageso))

    for filename, message in finalResult:
        print(filename, "contains", message)
        print("new itteration of for loop.")
    return finalResult

#Gör den zippad istället för lista av listor...
def analyseListOfMessages(sentimentResults, wordfeatures, voted_classifier):
    sentimentResultList = []
    for filename, message in sentimentResults:
        result = sentiment(message, voted_classifier, wordfeatures)
        print("Judgement and confidence of: ", filename,  result)
        temp = [filename, result]
        sentimentResultList.append(temp)
    return sentimentResultList

def printSentimentList(sentimentList):
    print("blablabla")
    for result in sentimentList:
        print(result[0], " classified as: ", result[1])

def trainAndPickleAllClassifiers(training_set):

    classifier = nltk.NaiveBayesClassifier.train(training_set)
    savePickle(classifier, "basicClassifier.pickle")
    print("Basic classifier saved")

    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    savePickle(MNB_classifier, "MNBClassifier.pickle")
    print("MNB classifier saved")

    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training_set)
    savePickle(BernoulliNB_classifier, "BNBClassifier.pickle")
    print("BNB classifier saved")

    LogisticRegression_classifier = SklearnClassifier(LogisticRegression(solver='liblinear'))
    LogisticRegression_classifier.train(training_set)
    savePickle(LogisticRegression_classifier, "LRClassifier.pickle")
    print("LRC classifier saved")

    SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
    SGDClassifier_classifier.train(training_set)
    savePickle(SGDClassifier_classifier, "SGDClassifier.pickle")
    print("SGD classifier saved")

    LinearSVC_classifier = SklearnClassifier(LinearSVC(max_iter=10000))
    LinearSVC_classifier.train(training_set)
    savePickle(LinearSVC_classifier, "LinearSVCClassifier.pickle")
    print("LinearSVC classifier saved")

    NuSVC_classifier = SklearnClassifier(NuSVC(gamma='auto'))
    NuSVC_classifier.train(training_set)
    savePickle(NuSVC_classifier, "NUSVCClassifier.pickle")
    print("NuSVC classifier saved")

def loadAndPickleDataset(posfile, negfile):

    featuresets = loadDatasetFromSingleFiles(posfile, negfile)
    trainingSet = featuresets[:10000]
    testingSet = featuresets[10000:]
    savePickle(featuresets, "featuresets.pickle")
    savePickle(trainingSet, "trainingset.pickle")
    savePickle(testingSet, "testingset.pickle")

def loadSwedishDataset(posfile, negfile):
    short_pos = open(posfile, "r").read()
    short_neg = open(negfile, "r").read()
    stop_words = list(set(stopwords.words('swedish')))

    documents = []
    all_words = []

    # J = Adjectives, this has to be changed for when the dataset is in swedish.
    #allowed_word_types = ["J"]

    for r in short_pos.split('\n'):
        documents.append((r, "pos"))
        words = word_tokenize(r)
        stopped = [w for w in words if not w in stop_words]

        pos = nltk.pos_tag(words)
        for w in pos:
            #if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

    for r in short_neg.split('\n'):
        documents.append((r, "neg"))
        words = word_tokenize(r)
        pos = nltk.pos_tag(words)
        for w in pos:
            #if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

    # short_pos_words = word_tokenize(short_pos)
    # short_neg_words = word_tokenize(short_neg)

    all_words = nltk.FreqDist(all_words)
    word_features = list(all_words.keys())[:5000]
    savePickle(word_features, "picklefiles_swe/word_features_swe.pickle")
    savePickle(documents, "picklefiles_swe/documents_swe.pickle")

    featuresets = [(find_features(rev, word_features), category) for (rev, category) in documents]
    random.shuffle(featuresets)
    return featuresets

def pickleAllSwedishClassifiers(basicC, NuSVC, BernouliC, LinearSVC, SGDC, MNBC, LRC):
    savePickle(basicC, "picklefiles_swe/basicClassifier_swe.pickle")
    savePickle(NuSVC, "picklefiles_swe/NuSVC_swe.pickle")
    savePickle(BernouliC, "picklefiles_swe/Bernouli_swe.pickle")
    savePickle(LinearSVC, "picklefiles_swe/LinearSVC_swe.pickle")
    savePickle(SGDC, "picklefiles_swe/SGD_classifier_swe.pickle")
    savePickle(MNBC, "picklefiles_swe/MultinomialNB_swe.pickle")
    savePickle(LRC, "picklefiles_swe/LogisticRegression_swe.pickle")

def translateMessageListToEnglish(messageList):
    translator = Translator()
    translatedMessagelist = []
    filenames = []
    for filename, message in messageList:
        translatedMessagelist.append(translator.translate(message, dest ='en', src='sv').text)
        filenames.append(filename)

    finalList = list(zip(filenames,translatedMessagelist))
    for filename, message in finalList:
        print(filename, "contains", message)

    return finalList

def loadTextfilesToList(directory):
    messages = []
    fileNames = []
    files = os.listdir(directory)
    for file in files:
        if(file.endswith(".txt")):
            fileNames.append(file)
            print(file)

    files = [open(directory + "/" + f, 'r').read() for f in files if not f.startswith('.')]
    for p in files:
        messages.append(p)

    finalList = list(zip(fileNames, messages))

    for filename, messageo in finalList:
        print(filename, "contains", messageo)
    return finalList