'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

# p(y=k) = 1/10
from sklearn.metrics import accuracy_score

alpha_y = 1/10
def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def add_pseudo_count(train_data, train_labels):
    # prior as pseudo count
    for i in range(10):
        i_array = np.array([i])
        train_labels = np.concatenate((train_labels, i_array), axis=0)
        train_labels = np.concatenate((train_labels, i_array), axis=0)

        ones = np.ones((1,64))
        zeros = np.zeros((1,64))

        train_data = np.concatenate((train_data, ones), axis=0)
        train_data = np.concatenate((train_data, zeros), axis=0)

    return train_data, train_labels

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''

    eta = np.zeros((10, 64))

    train_data, train_labels = add_pseudo_count(train_data, train_labels)

    for i in range(10):
        i_digits = train_data[train_labels == i]
        i_num_obsvervations = i_digits.shape[0]

        for j in range(64):
            j_column = i_digits[:,j]
            num_ones = np.sum(j_column == 1)

            eta[i, j] = num_ones / i_num_obsvervations

    return eta

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    img_array = []
    for i in range(10):
        img_i = class_images[i]
        # ...
        img_i = np.reshape(img_i, (8, 8))

        img_array.append(img_i)

    all_concat = np.concatenate(img_array, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()


def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))

    for i in range(10):
        for j in range(64):
            generated_data[i, j] = np.random.binomial(1, eta[i,j], 1)

    #generated_data = binarize_data(eta)
    plot_images(generated_data)



def generate_column_likelihood(observation, n_eta):

    log_likelihood = 0
    d = observation.shape[0]

    for i in range(d):
        log_likelihood += observation[i] * np.log(n_eta[i])
        log_likelihood += (1 - observation[i]) * np.log((1 - n_eta[i]))

    return(log_likelihood)


def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''

    N = bin_digits.shape[0]
    d = bin_digits.shape[1]
    likelihood_matrix = np.zeros((N, 10))

    for i in range(10):
        likelihood_matrix[:, i] = np.apply_along_axis(generate_column_likelihood, 1, bin_digits, eta[i])


    return likelihood_matrix

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    class_likelihood_matrix = generative_likelihood(bin_digits, eta)
    posterior_matrix = class_likelihood_matrix + np.log(alpha_y)

    return posterior_matrix

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    N = bin_digits.shape[0]
    cond_likelihood = conditional_likelihood(bin_digits, eta)

    total_probability = 0
    for i in range(0, N):
        i_label = int(labels[i])
        total_probability += cond_likelihood[i, i_label]

    avg_cond = total_probability / N

    # Compute as described above and return
    return avg_cond

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    # Compute and return the most likely class
    digits_predict = np.argmax(cond_likelihood, axis=1)

    # Compute and return the most likely class
    return digits_predict

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # Evaluation
    plot_images(eta)

    generate_new_data(eta)


    # Average Conditional Log Probability
    print("Average Log Probability for Train Data:")
    train_avg_conditional = avg_conditional_likelihood(train_data,train_labels, eta)
    print('\t{}'.format(train_avg_conditional))

    print("Average Log Probability for Test Data: ")
    test_avg_condtional = avg_conditional_likelihood(test_data,test_labels, eta)
    print('\t{}'.format(test_avg_condtional))


    # accuracy score
    test_pred = classify_data(test_data, eta)
    print("Accuracy Report on Test Set:")
    print(accuracy_score(test_labels, test_pred))

    test_pred = classify_data(train_data, eta)
    print("Accuracy Report on Train Set:")
    print(accuracy_score(train_labels, test_pred))



if __name__ == '__main__':
    main()
