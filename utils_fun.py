from sklearn.metrics import (
                              ConfusionMatrixDisplay,
                              classification_report,
                              accuracy_score,
                              recall_score,
                              precision_score,
                              precision_recall_curve,
                              roc_curve,
                              auc,
                              RocCurveDisplay
                              )

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

import matplotlib.pyplot as plt

from mlxtend.plotting import plot_decision_regions


class MetricsUtils:
  def __init__(self, y_test, y_pred, y_train, x_train, x_test, estimator):
    self.y_test = y_test
    self.y_pred = y_pred
    self.x_train = x_train
    self.x_test = x_test
    self.estimator = estimator
    self.y_train = y_train

  def show_scores(self):
    print(f"Accuracy: {accuracy_score(self.y_test, self.y_pred):.2f}")
    print(f"Recall: {recall_score(self.y_test, self.y_pred):.2f}")
    print(f"Precision: {precision_score(self.y_test, self.y_pred):.2f}")

  def show_precision_recall(self):
    y_scores = self.estimator.predict_proba(self.x_train)[:, 1]
    precision, recall, threshold = precision_recall_curve(self.y_train, y_scores)

    plt.figure(figsize=(14, 7))

    plt.plot(threshold, precision[:-1], label='precision')
    plt.plot(threshold, recall[:-1], label='recall')
    # plt.axvline(x=0.5, color='black', linestyle='--')
    plt.legend()

    plt.show()

  def plot_roc_auc_curve(self):
    y_pred_proba = self.estimator.predict_proba(self.x_test)[:, 1]
    fpr, tpr, thr = roc_curve(self.y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.show()

  def plot_margin(self, model_name, scaler_name):
    x = self.x_train
    y = self.y_train
    x_test = self.x_test
    estimator = self.estimator[model_name]
    scaler = self.estimator[scaler_name]

    fig, ax = plt.subplots(figsize=(10, 7))
    plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='autumn')

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    lin_x = np.linspace(xlim[0], xlim[1], 30)
    lin_y = np.linspace(ylim[0], ylim[1], 30)

    grid_Y, grid_X = np.meshgrid(lin_y, lin_x)

    xy = np.vstack([grid_X.ravel(), grid_Y.ravel()]).T

    xy = scaler.transform(xy)

    Z = estimator.decision_function(xy).reshape(grid_X.shape)

    ax.contour(grid_X, grid_Y, Z,
              colors='k',
              levels=[-1, 0, 1],
              alpha=0.5,
              linestyles=['--', '-', '--']
              )

    support_vectors = estimator.support_vectors_

    support_vectors = scaler.inverse_transform(support_vectors)

    ax.scatter(support_vectors[:, 0], support_vectors[:, 1],
              s=100, linewidth=3, facecolors='none', edgecolors='k')

    ax.scatter(x_test[:, 0], x_test[:, 1], marker='x', c=self.y_test)

    plt.show()

  def plot_boundary(self):
    plt.figure(figsize=(10, 7))
    plot_decision_regions(self.x_train,
                          self.y_train,
                          clf=self.estimator,
                          legend=2)
    plt.scatter(self.x_test[:, 0],
                self.x_test[:, 1],
                marker='x',
                c=self.y_test)
    plt.show()


class NotAlone(TransformerMixin, BaseEstimator):
    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=True)

        self.n_features_ = X.shape[1]
        return self

    def transform(self, X):
        X = check_array(X, accept_sparse=True)
        return X.sum(axis=1).astype(bool).astype(int).reshape(-1, 1)