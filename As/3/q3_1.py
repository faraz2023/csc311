'''
Question 3.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        distances = self.l2_distance(test_point)
        k_least_dist = distances.argsort()[0:k]
        digit = round(self.train_labels[k_least_dist].mean())
        return digit

    def predict(self, k, eval_data):
        eval_pred = np.zeros(eval_data.shape[0])
        for i in range(eval_data.shape[0]):
            eval_pred[i] = self.query_knn(eval_data[i], k)

        return eval_pred

def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    fold_number = 10

    best_accuracy = 0.0
    best_k = 0
    for k in k_range:
        kf = KFold(n_splits=fold_number)
        i = 1
        k_accuracy = 0.0
        for train_index, validate_index in kf.split(train_data):
            print("Running fold {} for k: {}".format(i, k))
            i += 1
            fold_train_data = train_data[train_index]
            fold_train_label = train_labels[train_index]
            fold_validate_data =  train_data[validate_index]
            fold_validate_label = train_labels[validate_index]
            #print(validate_index)
            fold_knn = KNearestNeighbor(fold_train_data, fold_train_label)
            fold_accuracy = classification_accuracy(fold_knn, k, fold_validate_data, fold_validate_label)
            print('\t{}'.format(fold_accuracy))
            k_accuracy += fold_accuracy

        k_accuracy = k_accuracy / fold_number
        if(k_accuracy > best_accuracy):
            best_accuracy = k_accuracy
            best_k = k

    return best_accuracy, best_k


def validate_results(y_eval, y_test):
    return  accuracy_score(y_test, y_eval) * 100

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''

    eval_pred = knn.predict(k, eval_data)
    return validate_results(eval_pred, eval_labels)


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data', shuffle=True)
    knn = KNearestNeighbor(train_data, train_labels)

    # sub part 1
    
    print("For K = 1 Classification Accuracy: {}".format(classification_accuracy(knn, 1, test_data, test_labels)))
    print("For K = 15 CLassification Accuracy: {}".format(classification_accuracy(knn, 15, test_data, test_labels)))
    
    
    #sub part 3
    print(cross_validation(train_data, train_labels))



    '''
    # Example usage:
    #for i in test_data:
    #    print(i.shape)
    j = 0
    #print(test_labels.shape[0])
    for i in range(10):
        #print(str(test_data[i]) + '  : ' + str(test_labels[i]))
        predicted_label = knn.query_knn(test_data[i], 20)
        print(predicted_label)
        print(test_labels[i])
        print('-----')

    for i in range(test_labels.shape[0]):
        predicted_label = knn.query_knn(test_data[i], 15)

        if(predicted_label !=  test_labels[i]):
            j += 1

    print(j)
    '''


if __name__ == '__main__':
    main()