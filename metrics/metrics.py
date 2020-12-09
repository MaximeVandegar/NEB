import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as sklearn_roc_auc_score


def roc_auc_score(predictions, targets):
    """
    Computes a classifier's performance in terms of ROC AUC
    
    Arguments:
    -------
        - predictions: classifier's score of the class with the greater label
        - targets
    """
    return sklearn_roc_auc_score(targets, predictions)


def compute_roc_auc(x0, x1, random_state=0, max_iter=500):
    """
    Given two datasets x0 and x1, trains a binary classifier to discriminate between them, and
    reports its performance in terms of ROC AUC.
    
    Inspired from https://github.com/gpapamak/snl/blob/master/inference/diagnostics/two_sample.py
    """

    labels = np.hstack([np.zeros(x0.shape[0]), np.ones(x1.shape[0])])
    x_train, x_test, y_train, y_test = train_test_split(np.vstack([x0, x1]), labels, stratify=labels,
                                                        random_state=random_state)

    # train a classifier
    classifier = MLPClassifier(hidden_layer_sizes=(x0.shape[1] * 10, x0.shape[1] * 10),
                               batch_size=100, max_iter=max_iter, solver='sgd').fit(x_train, y_train)

    return roc_auc_score(classifier.predict_proba(x_test)[:, 1], y_test)
