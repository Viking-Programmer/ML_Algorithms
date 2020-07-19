"""
Implement KNN on Iris Dataset w/o using library
"""

import pandas as pd

column_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
data = pd.read_csv("iris_dataset_knn.csv", names = column_names)

# replace class labels with integer values
data['class'].replace({'Iris-virginica':0, 'Iris-setosa':1, 'Iris-versicolor':2}, inplace = True)

# calculate euclidean distance between two row vectors
def euclidean_distance(vec1, vec2):
    sum_of_distance = 0
    for i in range(len(vec1) - 1):
        sum_of_distance += (vec1[i] - vec2[i])**2
    return (sum_of_distance)*0.5

def get_nearest_neighbors(training_dataset, test_row, k):
    all_distances = list()
    for i in range(len(training_dataset)):
        train_row = list(training_dataset.iloc[i,:])
        distance = euclidean_distance(train_row, test_row)
        all_distances.append((train_row, distance))
    all_distances.sort(key = lambda tup:tup[1])
    neighbors = list()
    for i in range(k):
        neighbors.append(all_distances[i][0])
    return neighbors

def predict_class(training_dataset, test_row, k):
    neighbors = get_nearest_neighbors(training_dataset, test_row, k)
    output_class = [neighbor[-1] for neighbor in neighbors]
    predicted_class = max(set(output_class), key = output_class.count)
    return predicted_class

test_row = [3.5, 4.5, 1.4, 0.9]
k = 10
label = predict_class(data, test_row, k)
print("Predicted label is %s." %label)