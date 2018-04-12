import classify
import numpy
import scipy
import util
import sklearn

def expand_data(speech):
    unlabeledX = speech.unlabeledX
    trainX = speech.trainX
    trainy = speech.trainy

    unlabeledX = sklearn.utils.shuffle(unlabeledX)

    total_unlabeled_count = unlabeledX.shape[0]

    best_clf = None
    best_acc = 0
    best_i = 0
    unlabeled_results = dict()
    n_samples = 100
    n_iterations = int(total_unlabeled_count/n_samples)

    print("Doing ", n_iterations, " iterations, with a sample size of ", n_samples)

    for i in range(n_iterations):
        clf = classify.train_classifier(trainX, trainy)

        # acc_before = evaluate(trainX, trainy, clf)

        newX = unlabeledX[:n_samples]
        unlabeledX = unlabeledX[n_samples:]
        newy = clf.predict(newX)

        trainX = scipy.sparse.vstack( [trainX, newX] )
        trainy = numpy.concatenate( [trainy, newy] )

        acc = classify.evaluate(speech.devX, speech.devy, clf)

        unlabeled_results[ (i+1)*n_samples ] = acc

        if acc > best_acc:
            best_acc = acc
            best_clf = clf
            best_i = i

        print("Iteration: ", i, " Accuracy: ", acc)
    
    util.print_dict_tofile(unlabeled_results)
    print("Best accuracy: ", best_acc, " samples of unlabeled data used",
            (best_i+1)*n_samples)

    return best_clf
