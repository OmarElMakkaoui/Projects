import numpy as np
from sklearn.model_selection import KFold
from my_LDA import my_LDA
from predict import predict


np.random.seed(1)
my_data = np.genfromtxt('wine_data.csv', delimiter=',')
np.random.shuffle(my_data)

X = my_data[:, 1:] 
y = my_data[:, 0]   


K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=1)

accuracies = []

for train_index, test_index in kf.split(X):
    trainingData, testData = X[train_index], X[test_index]
    trainingLabels, testLabels = y[train_index], y[test_index]
    
    # Training the LDA classifier
    W, projected_centroid, X_lda = my_LDA(trainingData, trainingLabels)
    
    # predictions for the test data
    predictedLabels = predict(testData, projected_centroid, W)
    predictedLabels = predictedLabels + 1  
    
    # accuracy
    counter = np.sum(predictedLabels == testLabels)
    accuracy = (counter / float(predictedLabels.size)) * 100.0
    accuracies.append(accuracy)


print(f'Average accuracy of LDA across {K}-fold cross-validation: {np.mean(accuracies):.2f}%')
