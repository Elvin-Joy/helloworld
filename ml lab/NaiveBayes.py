#7
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,f1_score

train_data=fetch_20newsgroups(
    subset="train",
    remove=("headers","footers","quotes")
)

test_data=fetch_20newsgroups(
    subset="test",
    remove=("headers","footers","quotes")
)

vectorizer=CountVectorizer(
    stop_words="english",
    max_features=20000
)

X_train=vectorizer.fit_transform(train_data.data)
X_test=vectorizer.transform(test_data.data)

y_train=train_data.target
y_test=test_data.target

model1=MultinomialNB()
model1.fit(X_train,y_train)
pred1=model1.predict(X_test)


print ("MULTINOMIAL NAIVE BAYES")
print("Accuracy",accuracy_score(y_test,pred1))
print ("F1 Score ",f1_score(y_test,pred1,average="macro"))


model2 = BernoulliNB()
model2.fit(X_train, y_train)
pred2 = model2.predict(X_test)

print("\nBernoulli Naive Bayes")
print("Accuracy:", accuracy_score(y_test, pred2))
print("F1-score:", f1_score(y_test, pred2,average="macro"))