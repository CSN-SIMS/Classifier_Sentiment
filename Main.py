#Lukas and Jenjira

from Functions import *

#Att göra: Se till att inladdningen av data från filer är korrekt, Klassifiera neutral baserat på confidence.

#Used to load training och testingsets.
documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)
testfeatures = loadPickleFile("testfeatures.pickle")
featuresets = loadPickleFile("features.pickle")
#word_features = loadPremadeMovieReviews()
#featuresets = [(find_features(rev, word_features), category) for (rev, category) in documents]
training_set = featuresets[:2000]
testing_set = testfeatures[:2000]

#Defining all classifiers to be used in the vote classifier.
classifier = nltk.NaiveBayesClassifier.train(training_set)
MNB_classifier = SklearnClassifier(MultinomialNB())
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
LogisticRegression_classifier = SklearnClassifier(LogisticRegression(solver='liblinear'))
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
LinearSVC_classifier = SklearnClassifier(LinearSVC(max_iter=10000))
NuSVC_classifier = SklearnClassifier(NuSVC(gamma='auto'))

print("Amount of documents in training set: ", len(training_set))
print("Amount of documents in testing set: ", len(testing_set))
voted_classifier = printAndTrainClassifiers(training_set, testing_set, classifier, MNB_classifier, BernoulliNB_classifier, LogisticRegression_classifier, SGDClassifier_classifier, LinearSVC_classifier, NuSVC_classifier)
printVoteConfidence(voted_classifier, testing_set)