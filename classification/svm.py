import os
import pickle
from sklearn import svm, cross_validation
from classification.metrics import show_confusion_matrix, print_classification_report
from image.feature_extraction import load_data


def train(clf, X_train, y_train):
    """ Train and return an SVM classifier """

    # TODO: train the classifier

    # TODO: return the classifier
    return None


def load_or_train(force_train=False):
    """
    Load an existing one or train a new SVM classifier, and return it.
    Once the classifier is trained, it is saved through pickle.
    """

    clf_path='./clf.pkl'
    data_path = "././data/ocr_data/"

    clf = None

    if not force_train and os.path.exists(clf_path):
        clf = pickle.load(open(clf_path, 'rb'))
    else:
        # Loading all data
        X, y = load_data(data_path)
        kernelList =['linear','rbf','poly','sigmoid']
        #kernelList =['linear','rbf','poly','sigmoid']
        kernelSelected=''
        kernelScore = -1000
        scores = None
        # Instantiate a classifier
        # TODO: instantiate a new classifier (choose an adapted kernel!)
        for oneKernel in kernelList :
            print("Cross valid kernel :"+oneKernel)
            clf = svm.SVC(kernel=oneKernel,C=1.0)
            # Cross validation
            # TODO: do cross-validation and print cross-validation result (mean accuracy +/- standard deviation)
            kfolds = 10
    
            scores = cross_validation.cross_val_score(clf, X, y, cv=kfolds)
            if scores.mean() > kernelScore :
                kernelSelected = oneKernel
                kernelScore = scores.mean()
                
            print "Score : "+"%.3f" % scores.mean()
        
        print "Score selected : "+ "%.3f" % scores.mean()
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print("Kernel selected : "+kernelSelected)
        # Train the classifier on the whole dataset, and save it
        # TODO: train the classifier on the whole dataset
        clf = svm.SVC(kernel=kernelSelected,C=1.0)
        
        clf.fit(X,y)
        pickle.dump(clf, open(clf_path, 'wb'))

        # If you want, you can do validation, print classification report and show confusion matrix with this
        # trained classifier. But keep in mind that you will do it on the training set itself!

    # TODO: return the classifier
    return clf