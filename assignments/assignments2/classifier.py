import numpy as np
from pprint import pprint
import pandas as pd
import csv
import matplotlib.pyplot as plt

# set random seed
np.random.seed(7)

with open('curran-thomas-assgn2-part1.csv', 'r') as f:
    d = csv.reader(f, delimiter=',')
    data = [[float(x) for x in row[1:]] for row in d]
    Y = [i[0] for i in data]
    X = [i[1:] for i in data]
    dummy = [x.append(0) for x in X]


def training_split(inputs, outputs, train=.8, test=.2):
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
    z = np.dot(w, x)+w[-1]
    _sigmoid = 1 / (1 + np.exp(-1*z))
    return _sigmoid


def gradient_vector(y, yhat, x):
    l_ce_vector = [(yhat-y)*x_i for x_i in x]
    return l_ce_vector


def L_ce(y, yhat):
    loss = -1*((y*np.log(yhat))+(1-y)*(np.log(1-yhat)))
    return loss


def calc_theta(w, g, lr=.1):
    return [round(w[i] - (lr*g[i]), 5) for i in range(0, len(g))]


def sgd(inputs, outputs, lr=.1, n_epochs=100):

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
coef, losses = sgd(xtrain, ytrain)

predictions = [[round(sigmoid(coef, xtest[i])), ytest[i], round(
    sigmoid(coef, xtest[i])) == ytest[i]] for i in range(0, len(xtest))]

df = pd.DataFrame(predictions, columns=['predicted', 'actual', 'correct'])

np.mean(df['correct'])
