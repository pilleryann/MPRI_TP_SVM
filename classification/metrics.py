from sklearn.metrics.metrics import confusion_matrix, classification_report
import pylab as pl


def show_confusion_matrix(y_true, y_predicted, title=''):
    """
    Plot (and print) a confusion matrix from y_true and y_predicted
    """

    # TODO: show confusion matrix plot
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_predicted)
    
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

    print("Report :"+title)
    target_names =['0','1','2','3','4','5','6','7','8','9']
    classification_report(y_true,y_pred,target_names=target_names)