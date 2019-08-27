from sklearn.datasets import load_files
import numpy as np
print("started...")
reviews_train = load_files("aclImdb/train/")
text_train, y_train = reviews_train.data, reviews_train.target

print("Number of documents in train data: {}".format(len(text_train)))
print("Samples per class (train): {}".format(np.bincount(y_train)))

reviews_test = load_files("aclImdb/test/")
text_test, y_test = reviews_test.data, reviews_test.target

print("Number of documents in test data: {}".format(len(text_test)))
print("Samples per class (test): {}".format(np.bincount(y_test)))


from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=5, ngram_range=(2, 2))
X_train = vect.fit(text_train).transform(text_train)
X_test = vect.transform(text_test)

aw=["i love to play football","i love to play cricket","strumming my paing with her fingers","love to"]
z=vect.transform(aw)
print(z)
print("Vocabulary size: {}".format(len(vect.vocabulary_)))
print("X_train:\n{}".format(repr(X_train)))
print("X_test: \n{}".format(repr(X_test)))

feature_names = vect.get_feature_names()
print("Number of features: {}".format(len(feature_names)))

xx=X_train[:100]
yy=y_train[:100]
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import svm
#param_grid = {'C': [.0001,.001,.01,.1, 0.9 , 1.0 ]}
param_grid = {'C': [ 0.9 , 1.0 ]}
svc = svm.SVC(gamma="scale")
grid = GridSearchCV(LogisticRegression(), param_grid, cv=6,verbose=1,n_jobs=4)
grid.fit(X_train, y_train)

print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)
print("Best estimator: ", grid.best_estimator_)



import matplotlib.pyplot as plt
import mglearn
mglearn.tools.visualize_coefficients(grid.best_estimator_.coef_, feature_names, n_top_features=5)
plt.show()


lr = grid.best_estimator_
lr.fit(X_train, y_train)
lr.predict(X_test)
print("Score: {:.2f}".format(lr.score(X_test, y_test)))

import pickle 
file=open("vector.pick","wb")
pickle.dump(vect,file)
file=open("classifier.pick","wb")
pickle.dump(lr,file)


pos=["this movie is not very good"]
print("Pos prediction: {}". format(lr.predict(vect.transform(pos))))
