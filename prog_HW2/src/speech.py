import os
import codecs

def read_files(data_loc):
    """Read the data files.
    The returned object contains various fields that store the data.
    The data is preprocessed for use with scikit-learn.
    
    Description of each fileds of returned object are as follows.
    
    - count_vec: CountVectorizer used to process the data (for reapplication on new data)
    - trainX,devX,testX,unlabeledX: Array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
    - le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
    - target_labels: List of labels (same order as used in le)
    - trainy,devy: Array of int labels, one for each document
    - test_fnames: List of test file names
    """

    class Data: pass
    speech = Data()

    print("-- train data")
    speech.train_data, speech.train_fnames, speech.train_labels = read_tsv(data_loc, "train.tsv")
    print(len(speech.train_data))

    print("-- dev data")
    speech.dev_data, speech.dev_fnames, speech.dev_labels = read_tsv(data_loc, "dev.tsv")
    print(len(speech.dev_data))

    print ("-- test data")
    speech.test_data, speech.test_fnames = read_unlabeled(data_loc,'test')
    print (len(speech.test_fnames))

    print("-- unlabeled data")
    speech.unlabeled_data, speech.unlabeled_fnames = read_unlabeled(data_loc, 'unlabeled')
    print(len(speech.unlabeled_fnames))

    print("-- transforming data and labels")
    speech = transform_data(speech, (1,2))

    from sklearn import preprocessing
    speech.le = preprocessing.LabelEncoder()
    speech.le.fit(speech.train_labels)
    speech.target_labels = speech.le.classes_
    speech.trainy = speech.le.transform(speech.train_labels)
    speech.devy = speech.le.transform(speech.dev_labels)
    return speech

def read_tsv(data_loc, fname):
    """Reads the labeled data described in tsv file.
    The returned object contains three fields that represent the unlabeled data.
    data: documents, each document is represented as list of words
    fnames: list of filenames, one for each document
    labels: list of labels for each document (List of string)
    """
    tf = codecs.open(data_loc + fname, 'r', encoding='utf-8')
    data = []
    labels = []
    fnames = []
    for line in tf:
        (ifname,label) = line.strip().split("\t")
        content = read_instance(data_loc, ifname)
        labels.append(label)
        fnames.append(ifname)
        data.append(content)
    tf.close()
    return data, fnames, labels

def read_unlabeled(data_loc, dname):
    """Reads the unlabeled data.
    The returned object contains two fields that represent the unlabeled data.
    data: documents, each document is represented as list of words
    fnames: list of filenames, one for each document
    """
    data = []
    fnames = []
    raw_fnames = os.listdir(data_loc+dname)
    for raw_fname in raw_fnames:
        fname = dname+'/'+ raw_fname
        content = read_instance(data_loc, fname)
        data.append(content)
        fnames.append(fname)
    return data, fnames

def read_instance(data_loc, ifname):
    """Reads the document file.
    Each document file has a string represents sequence of words, 
    and each words are seperated by semicolon.
    This function convert this string into list of words and return it.
    """
    inst = data_loc + ifname
    ifile = codecs.open(inst, 'r', encoding='utf-8')
    content = ifile.read().strip()
    content = content.split(';')
    return content

def write_pred_kaggle_file(cls, outfname, speech):
    """Writes the predictions in Kaggle format.

    Given the classifier, output filename, and the speech object,
    this function write the predictions of the classifier on the test data and
    writes it to the outputfilename. 
    """
    yp = cls.predict(speech.testX)
    labels = speech.le.inverse_transform(yp)
    f = codecs.open(outfname, 'w')
    f.write("FileIndex,Category\n")
    for i in range(len(speech.test_fnames)):
        fname = speech.test_fnames[i]
        f.write(fname + ',' + labels[i] + '\n')
    f.close()

def transform_data(speech, ngram_range=(1,1), use_tf=False):
    print("-- Transforming data...")

    if use_tf:
        from sklearn.feature_extraction.text import TfidfVectorizer
        speech.count_vect = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False,
                ngram_range=ngram_range)
    else:
        from sklearn.feature_extraction.text import CountVectorizer
        speech.count_vect = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False,
                ngram_range=ngram_range)

    import numpy
    speech.count_vect.fit(numpy.append(speech.train_data,speech.unlabeled_data))

    speech.trainX = speech.count_vect.transform(speech.train_data)
    speech.devX = speech.count_vect.transform(speech.dev_data)
    speech.testX = speech.count_vect.transform(speech.test_data)
    speech.unlabeledX = speech.count_vect.transform(speech.unlabeled_data)


    return speech

if __name__ == "__main__":
    print("Reading data")
    data_loc = "../data/"
    speech = read_files(data_loc)

    # best_acc = 0
    # best_ngram = (1,1)
    # ngram_results = dict()
    # # for ngram_range in [(1,1), (1,2), (1,3), (1,4), (2,2), (2,3), (2,4)]:
    # for ngram_range in [(1,2)]:
    #     speech = transform_data(speech, ngram_range)

    #     # for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]:
    #     for C in [1]:
    #         print("Training classifier")
    #         import classify
    #         cls = classify.train_classifier(speech.trainX, speech.trainy, C)

    #         print("Evaluating")
    #         # classify.evaluate(speech.trainX, speech.trainy, cls)
    #         acc = classify.evaluate(speech.devX, speech.devy, cls)
    #         if acc > best_acc:
    #             best_acc = acc
    #             best_ngram = ngram_range
    #         ngram_results[ngram_range] = acc

    # print("Best accuracy: ", best_acc)
    # print("Best ngram range: ", best_ngram)

    # import util
    # util.print_dict(ngram_results)

    import experiments
    cls = experiments.expand_data(speech)

    print("Writing Kaggle pred file")
    write_pred_kaggle_file(cls, data_loc + "/speech-pred.csv", speech)

