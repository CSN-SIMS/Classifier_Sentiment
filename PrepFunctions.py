import os
import pickle
import random
import nltk
from googletrans import Translator
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import LinearSVC, NuSVC
from nltk.classify.scikitlearn import SklearnClassifier


def pickleAllSwedishClassifiers(basicC, NuSVC, BernouliC, LinearSVC, SGDC, MNBC, LRC):
    savePickle(basicC, "picklefiles_swe/basicClassifier_swe.pickle")
    savePickle(NuSVC, "picklefiles_swe/NuSVC_swe.pickle")
    savePickle(BernouliC, "picklefiles_swe/Bernouli_swe.pickle")
    savePickle(LinearSVC, "picklefiles_swe/LinearSVC_swe.pickle")
    savePickle(SGDC, "picklefiles_swe/SGD_classifier_swe.pickle")
    savePickle(MNBC, "picklefiles_swe/MultinomialNB_swe.pickle")
    savePickle(LRC, "picklefiles_swe/LogisticRegression_swe.pickle")

def savePickle(saveObj, filename):
    save_obj = open(filename, "wb")
    pickle.dump(saveObj, save_obj)
    save_obj.close()

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

#Testingfunktion för att ladda och spara
def trainAndPickleAllClassifiers(training_set):

    classifier = nltk.NaiveBayesClassifier.train(training_set)
    savePickle(classifier, "picklefiles_eng/basicClassifier.pickle")
    print("Basic classifier saved")

    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    savePickle(MNB_classifier, "picklefiles_eng/MNBClassifier.pickle")
    print("MNB classifier saved")

    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training_set)
    savePickle(BernoulliNB_classifier, "picklefiles_eng/BNBClassifier.pickle")
    print("BNB classifier saved")

    LogisticRegression_classifier = SklearnClassifier(LogisticRegression(solver='liblinear'))
    LogisticRegression_classifier.train(training_set)
    savePickle(LogisticRegression_classifier, "picklefiles_eng/LRClassifier.pickle")
    print("LRC classifier saved")

    SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
    SGDClassifier_classifier.train(training_set)
    savePickle(SGDClassifier_classifier, "picklefiles_eng/SGDClassifier.pickle")
    print("SGD classifier saved")

    LinearSVC_classifier = SklearnClassifier(LinearSVC(max_iter=10000))
    LinearSVC_classifier.train(training_set)
    savePickle(LinearSVC_classifier, "picklefiles_eng/LinearSVCClassifier.pickle")
    print("LinearSVC classifier saved")

    NuSVC_classifier = SklearnClassifier(NuSVC(gamma='auto'))
    NuSVC_classifier.train(training_set)
    savePickle(NuSVC_classifier, "picklefiles_eng/NUSVCClassifier.pickle")
    print("NuSVC classifier saved")

#Error hantering här för när internet är nere.
def translateMessageListToEnglish(messageList):
    translator = Translator()
    translatedMessagelist = []
    filenames = []
    for filename, message in messageList:
        translatedMessagelist.append(translator.translate(message, dest ='en', src='sv').text)
        filenames.append(filename)

    finalList = list(zip(filenames,translatedMessagelist))
    return finalList

def loadTextfilesToList(directory):
    messages = []
    fileNames = []
    category = (directory.split('/')[1])
    files = os.listdir(directory)
    for file in files:
        if(file.endswith(".txt")):
            temp = [file, category]
            fileNames.append(temp)

    files = [open(directory + "/" + f, 'r').read() for f in files if not f.startswith('.')]
    for p in files:
        p = p.replace('\n', " ")
        messages.append(p)
    finalList = list(zip(fileNames, messages))

    return finalList

def loadMultipleDirectoriesToOneList(parentDirectory):
    returnList = []
    childDirectories = os.listdir(parentDirectory)
    for childDirectory in childDirectories:
        if not(childDirectory.startswith('.')):
            pathToDir = parentDirectory + "/" + childDirectory
            returnList += loadTextfilesToList(pathToDir)
    return returnList



