#Lukas and Jenjira

from Functions import *
#Att göra: Se till att inladdningen av data från filer är korrekt, hitta svenskt dataset, ladda in och klassa texter på ett effektivt sätt (icke träning eller testingset).

#Used to load training och testingsets.


#word_features = loadPremadeMovieReviews()
#featuresets = [(find_features(rev, word_features), category) for (rev, category) in documents]
#featuresets = loadDatasetFromSingleFiles("positive.txt", "negative.txt")
#training_set = featuresets[:10000]
#testing_set = featuresets[10000:]


training_set = loadPickleFile("trainingset.pickle")
testing_set = loadPickleFile("testingset.pickle")
document = loadPickleFile("documents.pickle")
word_features = loadPickleFile("word_features.pickle")

print("Size of trainingset: ", len(training_set))
print("Size of testingset: ", len(testing_set))
""""
savePickle(training_set, "trainingset.pickle")
savePickle(testing_set, "testingset.pickle")
#Defining all classifiers to be used in the vote classifier.

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
"""
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

#voted_classifier = printAndTrainClassifiers(training_set, testing_set, classifier, MNB_classifier, BernoulliNB_classifier, LogisticRegression_classifier, SGDClassifier_classifier, LinearSVC_classifier, NuSVC_classifier)
#printVoteConfidence(voted_classifier, testing_set)
messages = loadTextfilesForAnalysis("textfiles")
print(len(messages))
sentimentResults = analyseListOfMessages(messages, word_features, voted_classifier)
printSentimentList(sentimentResults)