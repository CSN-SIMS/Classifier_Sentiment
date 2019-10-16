#Lukas and Jenjira

from Functions import *
#Att göra: Ta bort stopwords och oönskade ordtyper. Hitta bra sätt att presentera statistik. Skapa ett relativt enkelt sätt att byta mellan svenska och engelska (Kind of done).
#Lägg till neutral klassifiering när testning är klar, Motverka neg klassifiering av texter som är osäkra. Lägg till bättre alternativ för inladdning av filer i flera directories.
#teckenkodning, t.ex utf-8 ren text osv...


#Inladdning av
#featuresets = loadDatasetFromSingleFiles("positive.txt", "negative.txt")
#training_set = featuresets[:10000]
#testing_set = featuresets[10000:]

""""
training_set = loadPickleFile("trainingset.pickle")
testing_set = loadPickleFile("testingset.pickle")
document = loadPickleFile("documents.pickle")
word_features = loadPickleFile("word_features.pickle")
"""

word_features = loadPickleFile("picklefiles_eng/word_features.pickle")
document = loadPickleFile("picklefiles_eng/documents.pickle")
training_set = loadPickleFile("picklefiles_eng/trainingset.pickle")
testing_set = loadPickleFile("picklefiles_eng/testingset.pickle")

print("Lenght of trainingset:", len(training_set))
print("Lenght of testingset:", len(testing_set))

classifier = loadPickleFile("picklefiles_eng/basicClassifier.pickle")
NuSVC_classifier = loadPickleFile("picklefiles_eng/NUSVCClassifier.pickle")
BernoulliNB_classifier = loadPickleFile("picklefiles_eng/BNBClassifier.pickle")
LinearSVC_classifier = loadPickleFile("picklefiles_eng/LinearSVCClassifier.pickle")
SGDClassifier_classifier = loadPickleFile("picklefiles_eng/SGDClassifier.pickle")
MNB_classifier = loadPickleFile("picklefiles_eng/MNBClassifier.pickle")
LogisticRegression_classifier = loadPickleFile("picklefiles_eng/LRClassifier.pickle")

print("Basic classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(50)


voted_classifier = VoteClassifier(classifier,
                                      NuSVC_classifier,
                                      BernoulliNB_classifier,
                                      LinearSVC_classifier,
                                      SGDClassifier_classifier,
                                      MNB_classifier,
                                      LogisticRegression_classifier)
""""
print("Multinomial Naive Bayes accuracy:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set)) * 100)
"""

messageList = loadTextfilesToList("textfiles")
translatedMessageList = translateMessageListToEnglish(messageList)
analyseListOfMessages(translatedMessageList, word_features, voted_classifier)


""""
print("Size of trainingset: ", len(training_set))
print("Size of testingset: ", len(testing_set))

savePickle(training_set, "trainingset.pickle")
savePickle(testing_set, "testingset.pickle")

#Load all Scikit-learn classifiers from previously saved pickle files.
classifier = loadPickleFile("basicClassifier.pickle")
NuSVC_classifier = loadPickleFile("NUSVCClassifier.pickle")
BernoulliNB_classifier = loadPickleFile("BNBClassifier.pickle")
LinearSVC_classifier = loadPickleFile("LinearSVCClassifier.pickle")
SGDClassifier_classifier = loadPickleFile("SGDClassifier.pickle")
MNB_classifier = loadPickleFile("MNBClassifier.pickle")
LogisticRegression_classifier = loadPickleFile("LRClassifier.pickle")


print("Basic classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
print("Multinomial Naive Bayes accuracy:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

voted_classifier = VoteClassifier(classifier,
                                      NuSVC_classifier,
                                      BernoulliNB_classifier,
                                      LinearSVC_classifier,
                                      SGDClassifier_classifier,
                                      MNB_classifier,
                                      LogisticRegression_classifier)
print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set)) * 100)

print(sentiment("This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea", voted_classifier, word_features))
print(sentiment("This movie was bad, would not watch again 0/10, not even a single mention of the overpopulation of otters", voted_classifier, word_features))
print(sentiment("Dear CSN, i am very happy with your services and appriciate the loan that you have given me, thanks", voted_classifier, word_features))
print(sentiment("Hello CSN, i am very dissatisfied with your services and i would not recommend you to my friends...", voted_classifier, word_features))

#voted_classifier = printAndTrainClassifiers(training_set, testing_set, classifier, MNB_classifier, BernoulliNB_classifier, LogisticRegression_classifier, SGDClassifier_classifier, LinearSVC_classifier, NuSVC_classifier)
#printVoteConfidence(voted_classifier, testing_set)
messages = loadTextfilesForAnalysis("textfiles")
print(len(messages))
sentimentResults = analyseListOfMessages(messages, word_features, voted_classifier)
printSentimentList(sentimentResults)
"""