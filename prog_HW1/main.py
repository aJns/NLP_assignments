# A naive bayes sentiment classifier
import sklearn.datasets
import sklearn.feature_extraction
import sklearn.naive_bayes
import sklearn.model_selection
import sklearn.metrics


def read_data(filepath):
    # read the data and transform it into a form that sklearn NB can use
    # use sklearn.datasets.load_files and sklearn.feature_extraction.text.CountVectorizer
    raw_data = sklearn.datasets.load_files(filepath, encoding="ASCII",
            load_content=True, shuffle=True)
    vectorizer = sklearn.feature_extraction.text.CountVectorizer()
    X_counts = vectorizer.fit_transform(raw_data.data)
    tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=False).fit(X_counts)
    X_tf = tf_transformer.transform(X_counts)
    y = raw_data.target

    return (X_tf, y, raw_data.target_names)


def train_nb(X_train, y_train):
    # return the data you need for classifying new instances
    clf = sklearn.naive_bayes.MultinomialNB().fit(X_train, y_train)
    return clf


def classify_nb(clf, X):
    # return the prediction of the classifier
    return clf.predict(X)


def main():
    filepath = "./data/txt_sentoken"
    X, y, target_names = read_data(filepath)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.10)
    nb_clf = train_nb(X_train, y_train)
    y_pred = classify_nb(nb_clf, X_test)

    clf_report = sklearn.metrics.classification_report(y_test, y_pred, target_names=target_names)
    print(clf_report)


if __name__ == "__main__":
    main()
