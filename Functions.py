#Lukas and Jenjira
#Created by following¨and making neacecary changes to a guide at https://pythonprogramming.net
#Work in progress
import os

import pickle
from openpyxl import load_workbook, Workbook
from nltk.tokenize import word_tokenize
from nltk.classify import ClassifierI
from statistics import mode
from collections import Counter

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

#Print all classifications and their confidence, ändra lidl...
def printVoteConfidence(voted_classifier, testingset):
    lidl = 0
    for i in testingset:
        print("Classification:", voted_classifier.classify(i[0]), "Confidence %:",voted_classifier.confidence(i[0])*100)

def find_features(document, wordfeatures):
    words = word_tokenize(document)
    features = {}
    for w in wordfeatures:
        features[w] = (w in words)
    return features

def sentiment(text, voted_classifier, wordfeatures):
    feats = find_features(text, wordfeatures)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)

#Används för att analysera en lista av medelanden inladdad av loadTextfilesToList och returnera en lista av filnamnen och [result, confidence]
def analyseListOfMessages(sentimentResults, wordfeatures, voted_classifier):
    sentimentResultList = []
    for filename, message in sentimentResults:
        result = sentiment(message, voted_classifier, wordfeatures)
        print("Judgement and confidence of: ", filename,  result)
        temp = [filename, result]
        sentimentResultList.append(temp)
    return sentimentResultList

#Testingfunktion för att printa en lista av filnamn och bedömningar
def printSentimentList(sentimentList):
    for result in sentimentList:
        print(result[0], " classified as: ", result[1])

def saveExcelFormat(judgementList, sheetname, percentages, filename, append = False):

    if(append):
        sentimentList = []
        wb = load_workbook(filename)
        ws = wb.get_sheet_by_name(sheetname)
        ws.cell(1, 1, "Filename")
        ws.cell(1, 2, "Category")
        ws.cell(1, 3, "Judgement %")
        ws.cell(1, 4, "Confidence")
        y = ws.max_row
        print(y)
        y += 1
        print(y)

        for fn, judgement in judgementList:
            ws.cell(y, 1, fn[0])
            ws.cell(y, 2, fn[1])
            ws.cell(y, 3, judgement[0])
            ws.cell(y, 4, judgement[1])
            y += 1

        wb.save(filename)
        print(ws.max_row)
        for i in range(2,ws.max_row+1):
            cellObj = (ws.cell(row=i, column=3))
            sentimentList.append(cellObj.value)
        percentages = updatePercentages(sentimentList)
        y=1
        for sent, conf in percentages:
            print(percentages[y-1][0])
            ws.cell(y,5, percentages[y-1][0] + " " + str(percentages[y-1][1]) + "%")
            y +=1

        wb.save(filename)


    else:
        wb = load_workbook(filename)
        ws = wb.create_sheet(sheetname, 0)
        ws.cell(1,1,"Filename")
        ws.cell(1,2, "Category")
        ws.cell(1,3, "Judgement %")
        ws.cell(1,4, "Confidence")
        y = 1
        for sent, conf in percentages:
            ws.cell(y,5, percentages[y-1][0] + " " + str(percentages[y-1][1]) + "%")
            y += 1
        y = 2
        for fn, judgement in judgementList:
            ws.cell(y,1, fn[0])
            ws.cell(y,2, fn[1])
            ws.cell(y,3, judgement[0])
            ws.cell(y,4, judgement[1] * 100)
            y +=1
        print("Max row:", ws.max_row)
        print("Max column:", ws.max_column)
        wb.save(filename)


def calculatePercentagesOfList(judgementList):
    newJudgementList = []
    for filename, judgement in judgementList:
        newJudgementList.append(judgement[0])

    c = Counter(newJudgementList)
    percentages = [(i, c[i] / len(newJudgementList) * 100) for i, count in c.most_common()]
    return percentages


def updatePercentages(totalJudgements):
    c = Counter(totalJudgements)
    percentages = [(i, c[i] / len(totalJudgements) * 100) for i, count in c.most_common()]
    print(percentages)
    return percentages

def saveToNewExcelfile(judgementList, sheetname, percentages, filename):
    wb = Workbook()
    ws = wb.create_sheet(sheetname, 0)
    ws.cell(1, 1, "Filename")
    ws.cell(1, 2, "Category")
    ws.cell(1, 3, "Judgement %")
    ws.cell(1, 4, "Confidence")
    y = 1
    for sent, conf in percentages:
        ws.cell(y, 5, percentages[y - 1][0] + " " + str(percentages[y - 1][1]) + "%")
        y += 1
    y = 2
    for fn, judgement in judgementList:
        ws.cell(y, 1, fn[0])
        ws.cell(y, 2, fn[1])
        ws.cell(y, 3, judgement[0])
        ws.cell(y, 4, judgement[1] * 100)
        y += 1
    print("Max row:", ws.max_row)
    print("Max column:", ws.max_column)
    wb.save(filename)