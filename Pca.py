from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.neural_network import MLPRegressor
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

X = pd.read_csv(f"Data\\{'Player'}.csv")
X = X.drop(['result', 'position', 'player',	'team'], axis=1).to_numpy()
y = pd.read_csv(f"Data\\{'Player'}.csv")['result'].to_numpy()

print(f'Starting Shape {X.shape}')

clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X, y)

print(f'ExtraTreeClassifier {clf.feature_importances_}')


# model = SelectFromModel(clf, prefit=True)
# X_new = model.transform(X)
# clf = Pipeline([
#     ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,max_iter=10000))),
#     ('classification', RandomForestClassifier())
# ])

# clf.fit(X, y)

# print(f'Pipeline {X_new.shape}')


lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)


print(f'LinearSVC {X_new.shape}')


X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
print(f'SelectKBest {X_new.shape}')


# # Create the RFE object and rank each pixel
# svc = SVC(kernel="linear", C=1)
# rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
# rfe.fit(X, y)


# print(f'SelectKBest {rfe.shape}')


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print(X_test)
regr = MLPRegressor(random_state=1, max_iter=10000).fit(X_train, y_train)
regr.predict(X_test)

print(regr.score(X_test, y_test))
