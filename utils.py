import numpy as np
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt


def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # load the data
    train_df = pd.read_csv('data/mnist_test.csv')
    test_df = pd.read_csv('data/mnist_train.csv')

    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values

    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values

    return X_train, X_test, y_train, y_test


def normalize(X_train, X_test) -> Tuple[np.ndarray, np.ndarray]:
    # normalize the data
    X_train = normalize_input(X_train)
    X_test = normalize_input(X_test)
    
    return (X_train,X_test)
    #raise NotImplementedError


def plot_metrics(metrics) -> None:
    # plot and save the results
    ks, accuracies, precisions, recalls, f1_scores = zip(*metrics)

    # create a 2x2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # plot the accuracy
    axs[0, 0].plot(ks, accuracies)
    axs[0, 0].set_xlabel('k')
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].set_title('Accuracy vs. k')

    # plot the precision
    axs[0, 1].plot(ks, precisions)
    axs[0, 1].set_xlabel('k')
    axs[0, 1].set_ylabel('Precision')
    axs[0, 1].set_title('Precision vs. k')

    # plot the recall
    axs[1, 0].plot(ks, recalls)
    axs[1, 0].set_xlabel('k')
    axs[1, 0].set_ylabel('Recall')
    axs[1, 0].set_title('Recall vs. k')

    # plot the F1 score
    axs[1, 1].plot(ks, f1_scores)
    axs[1, 1].set_xlabel('k')
    axs[1, 1].set_ylabel('F1 Score')
    axs[1, 1].set_title('F1 Score vs. k')

    # adjust the layout and show the plot
    fig.tight_layout()
    plt.savefig('metrics_plot.png')
    plt.show()
    
   
def normalize_input(x) -> np.ndarray:
    x = x/255
    x = 2*(x-0.5)
    
    return x
