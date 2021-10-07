import numpy as np
from pprint import pprint
import pandas as pd
import csv
import matplotlib.pyplot as plt

# set random seed
np.random.seed(7)

with open('part1/curran-thomas-assgn2-part1.csv', 'r') as f:
    d = csv.reader(f, delimiter=',')
    data = [[float(x) for x in row[1:]] for row in d]
    Y = [i[0] for i in data]
    X = [i[1:] for i in data]
    dummy = [x.append(0) for x in X]


def training_split(inputs, outputs, train=.8, test=.2):
    """[summary]

    Args:
        inputs ([2d array]): the set of features you want train your model on
        outputs ([1d array]): the set of outcomes to predict based on features
        train (float, optional): Proportion of inputs in dataset to include in training data Defaults to .8.
        test (float, optional): Proportion of outcomes dataset to include in the Defaults to .2.

    Returns:
        xtrain: 2d array of training data features
        ytrain: 1d array of outcomes to
        ytest: 1d array of outcomes to test
        xtest: 2d array of inputs to test model
    """
    if train + test != 1.0:
        return 'must add up to 1'

    train_n = int(round(len(inputs)*train))

    training_indexes = np.random.choice(
        np.arange(0, train_n+1), size=train_n, replace=False
    )

    training_feature_vals = []
    training_outcome_vals = []
    test_feature_vals = []
    test_outcome_vals = []

    for i in range(0, len(data)):
        if i in training_indexes:
            training_feature_vals.append(inputs[i])
            training_outcome_vals.append(outputs[i])
        else:
            test_feature_vals.append(inputs[i])
            test_outcome_vals.append(outputs[i])

    return training_feature_vals, training_outcome_vals, test_feature_vals, test_outcome_vals


def sigmoid(w, x):
    """[summary]
    Sigmoid function is used to predict binary outcome based on a set of features and weights.
    Args:
        w (list): set of weights associated with model to predict outcome
        x (list): set of features to test against weights to predict outcome

    Returns:
        float: returns singular sigmoid value
    """
    z = np.dot(w, x)+w[-1]
    _sigmoid = 1 / (1 + np.exp(-1*z))
    return _sigmoid


def gradient_vector(y, yhat, x):
    """[summary]

    * returns the gradient vector of losses to compute theta from previous weight vector
    Args:
        y (int): actual value of outcome based on data
        yhat (float): predicted value of outcome based on (x) features
        x (list): list of feature values from model

    Returns:
        list: returns the gradient of feature vector
    """
    l_ce_vector = [(yhat-y)*x_i for x_i in x]
    return l_ce_vector

def calc_theta(w, g, lr=.1):
    """[summary]
    Calculates the updated theta values based on the freshly calculated gradients for a given
    set of weights and features and predicted outcomes.

    Args:
        w (list): list of previous iteration's weights
        g (list): list of current interation's gradient values based on features and predicted outcomes
        lr (float, optional): learning rate to apply to list of gradients. Defaults to .1.

    Returns:
        [list]: returns list of updated weights to used for the next iteration of SGD
    """
    return [round(w[i] - (lr*g[i]), 5) for i in range(0, len(g))]


def sgd(inputs, outputs, lr=.1, n_epochs=100):
    """[summary]
    
    Performs stochastic gradient descent for a set of inputs and outputs

    Args:
        inputs (list]): 2d list of features to train stochastic gradient descent on
        outputs (list): list of 1d outcomes associated with each feature vector
        lr (float, optional): learning rate to be used in calculation of theta. Defaults to .1.
        n_epochs (int, optional): number of iterations. Defaults to 100.

    Returns:
        _w (list): returns the optimal weights for logistic model
        epoch_losses (list): returns the total loss for each epoch to be used for reporting
    """

    _w = [0] * len(inputs[0])
    _i = np.random.choice(np.arange(0, len(inputs)),
                          size=len(inputs), replace=False)
    epoch_losses = []
    for epoch in range(0, n_epochs):
        total_loss = 0
        for i in _i:
            _x = inputs[i]
            _y = outputs[i]
            _yhat = sigmoid(_w, _x)
            _loss = L_ce(_y, _yhat)

            total_loss += _loss

            _gradient_vector = gradient_vector(_y, _yhat, _x)
            _w = calc_theta(_w, _gradient_vector, lr)

        epoch_losses.append(total_loss)

    return _w, epoch_losses


xtrain, ytrain, xtest, ytest = training_split(X, Y)
coef, losses = sgd(X, Y)

#predictions = [[round(sigmoid(coef, X[i])), Y[i], round(
#    sigmoid(coef, X[i])) == Y[i]] for i in range(0, len(X))]

#df = pd.DataFrame(predictions, columns=['predicted', 'actual', 'correct'])

#print(np.mean(df['correct']))

#####################################################################################################################
# Implement Model on test data provided in canvas
#####################################################################################################################

with open('part1/assgn2-testset-reviews.csv', 'r') as f:
    d = csv.reader(f, delimiter=',')
    IDs = []
    Y = []
    X = []
    for row in d:
        IDs.append(row[0])
        Y.append(None)
        X.append([float(row[i]) for i in range(2, len(row))])
        
    for x in X:
        x.append(0)

outcomes = []
for i in range(0, len(X)):
    prediction = round(sigmoid(coef, X[i]))

    if prediction == 1:
        prediction_value = 'POS'
    else:
        prediction_value = 'NEG'

    outcomes.append([IDs[i], prediction_value])

pprint(outcomes)

with open('curran-thomas-assgn2.txt', 'w') as f:
    for value in outcomes:
        f.write(", ".join(value)+"\n")
    
