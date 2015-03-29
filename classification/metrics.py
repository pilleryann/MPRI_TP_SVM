from sklearn.metrics.metrics import confusion_matrix, classification_report
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    target_names =['0','1','2','3','4','5','6','7','8','9']
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def show_confusion_matrix(y_true, y_predicted, title=''):
    """
    Plot (and print) a confusion matrix from y_true and y_predicted
    """

    # TODO: show confusion matrix plot
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_predicted)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    print(cm)
    
    # Show confusion matrix in a separate window
    pl.matshow(cm)
    pl.title(title)
    pl.colorbar()
    pl.ylabel('True label')
    pl.xlabel('Predicted label')
    pl.show()


def print_classification_report(y_true, y_pred, title=''):
    """
    Print a classification report
    """

    # TODO: print classification report

    print(title)
    print(classification_report(y_true,y_pred))