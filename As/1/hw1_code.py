import math
import sys

from scipy.stats import entropy
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from sklearn import tree
import graphviz
from IPython.display import Image
from collections import Counter
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

np.set_printoptions(threshold=sys.maxsize)

from sklearn.model_selection import train_test_split

vectorizer = CountVectorizer()

"""part a"""
def load_data():

    #open, read, strip real_news
    real = open('clean_real.txt')
    real = real.readlines()
    real = [line.strip() for line in real]
    y = [True for i in range(len(real))]

    #open, read, strip fake news
    fake = open('clean_fake.txt')
    fake = fake.readlines()
    fake = [line.strip() for line in fake]
    y += [False for i in range(len(fake))]

    all_news = real + fake

    global vectorizer
    X = vectorizer.fit_transform(all_news)


    X_train, X_test, y_train, y_test \
        = train_test_split(X, y, test_size=0.3, random_state=42)

    X_test, X_val, y_test, y_val \
        = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_val': X_val,
        'y_train': y_train,
        'y_test': y_test,
        'y_val': y_val
    }

# this helper function creates a tree instace and trains it by the data given to it
def train_tree(X_train, y_train, max_depth, criterion):
    # Creating the classifier object
    clf = DecisionTreeClassifier(criterion=criterion,
                                      max_depth=max_depth)

    # Performing training
    clf.fit(X_train, y_train)
    return clf

# Function to make predictions by a model
def prediction(X_test, model):
    # Predicton on test with giniIndex
    y_pred = model.predict(X_test)
    return y_pred

# this function measures the accuracy of a model
def validate_model(model, X_test, y_test):
    y_pred = prediction(X_test, model)
    # print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
    # print("Report : " classification_report(y_test, y_pred))
    accurancy = accuracy_score(y_test, y_pred) * 100
    print("\tAccuracy : ",
      accurancy)

    return accurancy

""" part b"""
def select_tree_model(leaning_data):

    # I have choosen scaler multipications of log2(len(test_data) from 1 to 5 for "sensible" max depth
    trainig_size_log = int(math.log2(len(learning_data['X_train'].toarray())))

    max_accuracy = 0
    best_clf = None

    # tree trining
    for criterion in ['gini', 'entropy']:
        for i in range(1, 5):
            max_depth = int(1.5 * i * trainig_size_log)
            clf = train_tree(learning_data['X_train'], learning_data['y_train'], max_depth, criterion)
            print("Results for ", criterion, " index, max depth of ", max_depth, " :")
            accuracy = validate_model(clf, learning_data['X_val'], learning_data['y_val'])
            if(accuracy > max_accuracy):
                max_accuracy = accuracy
                best_clf = clf
    return max_accuracy, best_clf

def visualize_validate_matrix(validate_matrix):
    X = np.linspace(1, 20, 20, dtype=int)
    Y1 = validate_matrix[0, 1:21]
    Y2 = validate_matrix[1, 1:21]

    blue_patch = mpatches.Patch(color='blue', label='Training Set Accuracy')
    green_patch = mpatches.Patch(color='green', label='Validate Set Accuracy')

    plt.legend(handles=[blue_patch, green_patch])

    plt.xticks(X)
    plt.xlabel('K Neighbours')
    plt.ylabel('Accuracy Rate')
    plt.plot(X, Y1, color='blue')
    plt.plot(X, Y2, color='green')
    plt.show()



"""part e"""
def select_knn_model(learning_data):
    validate_matrix = np.zeros((2, 21), dtype=float)

    best_knn_model = None
    best_num_neighbour = 0
    best_accuracy = 0

    for num_neighbour in range(1, 21):
        knn_model = KNeighborsClassifier(n_neighbors=num_neighbour)
        knn_model.fit(learning_data['X_train'], learning_data['y_train'])
        print('Training Accuracy for {} Nearest Neighbours -->'.format(num_neighbour), end=" ")
        train_acc = validate_model(knn_model, learning_data['X_train'], learning_data['y_train'])
        print(r'Validation Accuracy for {} Nearest Neighbours -->'.format(num_neighbour), end=" ")
        validate_acc = validate_model(knn_model, learning_data['X_val'], learning_data['y_val'])
        validate_matrix[0, num_neighbour] = train_acc
        validate_matrix[1,num_neighbour] = validate_acc
        if(validate_acc > best_accuracy):
            best_knn_model = knn_model
            best_num_neighbour = num_neighbour
            best_accuracy = validate_acc

    print("------")
    print("Best model on validation set is with {} neighbours, with accuracy {}".format(best_num_neighbour, best_accuracy))
    print('Test Accuracy for {} Nearest Neighbours -->'.format(best_num_neighbour), end=" ")
    validate_model(best_knn_model, learning_data['X_test'], learning_data['y_test'])

    visualize_validate_matrix(validate_matrix)

    return best_accuracy, best_knn_model

"""part d"""
def compute_information_gain(X, y, x='donald'):

    contains_x = [(headline, isReal) for headline, isReal in zip(X, y) if x in vectorizer.inverse_transform(headline)[0]]
    not_contains_x = [(headline, isReal) for headline, isReal in zip(X, y) if not (x in vectorizer.inverse_transform(headline)[0])]

    entropy_y = entropy(list(Counter(y).values()), base=2)

    entropy_y_cond_x = 0
    for l in [contains_x, not_contains_x]:
        l = [item[1] for item in l]
        entropy_l =  entropy(list(Counter(l).values()), base=2)
        entropy_y_cond_x += len(l)/ len(y) * entropy_l
        #print(entropy_l)

    return entropy_y - entropy_y_cond_x


if __name__ == "__main__":

    # for trying part <a> run the following code
    learning_data = load_data()
    feature_names = vectorizer.get_feature_names()

    # for trying part <b> run the following code
    """
    accuracy, tree_model = select_tree_model(learning_data)
    """

    # # for trying part <c> run the following code
    """
    print("Max accuracy on validation sets: {}".format(accuracy))
    print("-----\n Best tree is based on: {} and gets this accuracy on the training set:".format(tree_model.get_params()['criterion']))
    validate_model(tree_model, learning_data['X_test'], learning_data['y_test'])
    
    dot_data = tree.export_graphviz(tree_model,feature_names=feature_names, filled=True, class_names=['Fake', 'Real'], max_depth=2)
    graph = graphviz.Source(dot_data)
    graph.render('my-tree')
    """

    # for trying part <d> run the following code
    """
    l = ['noneexistenceword', 'the', 'hillary', 'trumps', 'trump', 'donald', 'blasts', 'war']
    for word in l:
        ig = compute_information_gain(learning_data['X_train'], learning_data['y_train'], x=word)
        print("Information Gain for <{}> is: {}".format(word, ig))
    """

    # for trying part <e> run the following code
    """
    accuracy, knn_model = select_knn_model(learning_data)
    """









