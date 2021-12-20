import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

# Loading the data
train = pd.read_csv('data/train.csv', sep=',')
test = pd.read_csv('data/test.csv', sep=',')

# Convert data to List (sklearn requires lists)
trainingTexts = train['tweet'].tolist()
trainingTargets = train['sent'].tolist()
testTexts = test['tweet'].tolist()
testTargets = test['sent'].tolist()

# Apply Bag of Words
vect = CountVectorizer()
X_train = vect.fit_transform(trainingTexts).toarray()
X_test = vect.transform(testTexts).toarray()

# Model init and training
lr = LogisticRegression()
lr.fit(X_train, trainingTargets)

# Predictions on test
testPredict = lr.predict(X_test)

# Evaluate predictions
print("~~~Results~~~\n")
print(confusion_matrix(testTargets, testPredict))
print(f1_score(testTargets, testPredict, average='weighted'))
