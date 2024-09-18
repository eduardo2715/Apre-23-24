import matplotlib.pyplot as plt
from sklearn import metrics, datasets, tree
from sklearn.model_selection import train_test_split

# 1. load 
dt = datasets.load_digits() 
X, y = dt.data, dt.target

#from sklearn.datasets import fetch_openml
#mnist = fetch_openml('mnist_784')

# partition data with train_test_split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,stratify=y,random_state=36)


print("train size:",len(X_train),"\ntest size:",len(X_test))

# 2. learn classifier
from sklearn.linear_model import LogisticRegression

predictor = LogisticRegression(max_iter=10000) 
predictor.fit(X_train, y_train)

y_pred = predictor.predict(X_test)
print("accuracy on testing set:",  round(metrics.accuracy_score(y_test, y_pred),2))

# 2. learn classifier
from sklearn.neural_network import MLPClassifier

predictor = MLPClassifier(hidden_layer_sizes=(40,30),random_state=36,activation ='relu',solver='sgd')
#predictor = MLPClassifier(random_state=42)
predictor.fit(X_train, y_train)

y_pred = predictor.predict(X_test)
print("accuracy on testing set:",  round(metrics.accuracy_score(y_test, y_pred),4))

plt.plot(predictor.loss_curve_)
plt.title("Loss Curve", fontsize=14)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()
