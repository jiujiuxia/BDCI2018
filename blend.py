from __future__ import division
import numpy as np
import load_data
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC


def logloss(attempt, actual, epsilon=1.0e-15):
    """Logloss, i.e. the score of the bioresponse competition.
    """
    attempt = np.clip(attempt, epsilon, 1.0-epsilon)
    return - np.mean(actual * np.log(attempt) + (1.0 - actual) * np.log(1.0 - attempt))


if __name__ == '__main__':

    np.random.seed(0) # seed to shuffle the train set

    n_folds = 2
    verbose = True
    shuffle = False

    X, y, X_submission = load_data.load()

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]
    #print(y)
    skf = list(StratifiedKFold(y, n_folds))
    # print(skf)
    # print(type(skf))
    clfs = [XGBClassifier(learning_rate =0.5,n_estimators=200,max_depth=5,gamma=0,subsample=0.8,),
            RandomForestClassifier(n_estimators=200, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=200, n_jobs=-1, criterion='gini'),
            SVC(kernel='rbf', C=16, gamma=0.125, probability=True),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=100)
            ]

    print("Creating train and test sets for blending.")
    # print(X.shape[0])
    dataset_blend_train = np.zeros((X.shape[0], len(clfs),11))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs),11))

    dataset_blend_train_stack = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test_stack = np.zeros((X_submission.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        print(j, clf)
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf),11))
        for i, (train, test) in enumerate(skf):
            print("Fold", i)
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)
            #y_submission = clf.predict_proba(X_test)[:,1]
            # print('y_submission',y_submission)
            dataset_blend_train[test, j] = y_submission
            # print('dataset_blend_train',dataset_blend_train)
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)
            # print('test,j',dataset_blend_test_j)
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)
        # print('final test',dataset_blend_test)
        dataset_blend_train_stack[:,j] =  np.argmax(dataset_blend_train[:, j],axis=1)
        # print('train')
        # print(dataset_blend_train_stack)
        dataset_blend_test_stack[:, j] = np.argmax(dataset_blend_test[:, j], axis=1)
        # print('test')
        # print(dataset_blend_test_stack)




    print()
    print("Blending.")
    # clf = LogisticRegression()
    clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
    clf.fit(dataset_blend_train_stack, y)

    y_submission = clf.predict_proba(dataset_blend_test_stack)

    print('final')
    print(y_submission)
    #
    # print("Linear stretch of predictions to [0,1]")
    # y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    print("Saving Results.")
    np.savetxt(fname='test_new.csv', X=np.argmax(y_submission,axis = 1), fmt='%0.9f')
    np.savetxt(fname='multi_model_result_new.csv', X=dataset_blend_test_stack,fmt='%0.9f')
    
