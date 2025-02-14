import numpy as np
import pandas as pd
import pylab as plt

data = pd.read_csv('train.csv')
print("Dataset shape: ", data.shape)

binary_data = data[np.logical_or(data['Cover_Type'] == 1, data['Cover_Type'] == 2)]
X = binary_data.drop('Cover_Type', axis=1).values
y = binary_data['Cover_Type'].values
y = 2 * y - 3

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=8)
model.fit(X_train, y_train)

from sklearn.metrics import classification_report, accuracy_score
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=np.sort(data['Cover_Type'].unique().astype(str))))
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))

depth = 2
num_trees = 10
weights = np.ones(X_train.shape[0]) / X_train.shape[0]
train_scores = np.zeros(X_train.shape[0])
test_scores = np.zeros(X_test.shape[0])
train_errors, test_errors = [], []

for i in range(num_trees):
    weak_model = DecisionTreeClassifier(max_depth=depth)
    weak_model.fit(X_train, y_train, sample_weight=weights)
    y_pred_train = weak_model.predict(X_train)
    
    incorrect = np.not_equal(y_pred_train, y_train)
    gamma = weights[incorrect].sum() / weights.sum()
    alpha = np.log((1 - gamma) / gamma)
    
    weights *= np.exp(alpha * incorrect) 

    train_scores += alpha * y_pred_train
    train_error = np.mean(train_scores * y_train < 0)
    
    y_pred_test = weak_model.predict(X_test)
    test_scores += alpha * y_pred_test
    test_error = np.mean(test_scores * y_test < 0)

    train_errors.append(train_error)
    test_errors.append(test_error)

plt.plot(train_errors, label="Training error")
plt.plot(test_errors, label="Test error")
plt.legend()
plt.show()

def AdaBoost(depth, rounds):
    weights = np.ones(X_train.shape[0]) / X_train.shape[0]
    train_scores, test_scores = np.zeros(X_train.shape[0]), np.zeros(X_test.shape[0])
    train_errors, test_errors = [], []

    for i in range(rounds):
        weak_model = DecisionTreeClassifier(max_depth=depth)
        weak_model.fit(X_train, y_train, sample_weight=weights)
        y_pred_train = weak_model.predict(X_train)

        incorrect = np.not_equal(y_pred_train, y_train)
        gamma = weights[incorrect].sum() / weights.sum()
        alpha = np.log((1 - gamma) / gamma)
        weights *= np.exp(alpha * incorrect) 

        train_scores += alpha * y_pred_train
        train_error = np.mean(train_scores * y_train < 0)
        test_scores += alpha * weak_model.predict(X_test)
        test_error = np.mean(test_scores * y_test < 0)

        train_errors.append(train_error)
        test_errors.append(test_error)

    return test_error

depths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
final_test_errors = [AdaBoost(d, 100) for d in depths]

plt.plot(depths, final_test_errors)
plt.xlabel("Tree Depth")
plt.ylabel("Test Error")
plt.title("Test Error vs. Tree Depth")
plt.show()
