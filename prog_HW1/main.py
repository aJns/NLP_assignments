# A naive bayes sentiment classifier
import sklearn.datasets
import sklearn.feature_extraction


def read_data(filepath):
    # read the data and transform it into a form that sklearn NB can use
    # use sklearn.datasets.load_files and sklearn.feature_extraction.text.CountVectorizer
    raw_data = sklearn.datasets.load_files(filepath, encoding="ASCII",
            load_content=True, shuffle=True)
    vectorizer = sklearn.feature_extraction.text.CountVectorizer()
    X_counts = vectorizer.fit_transform(raw_data.data)
    tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=False).fit(X_counts)
    X_tf = tf_transformer.transform(X_counts)

    return (0,0)
    



def train_nb(training_documents):
    # ...
    # return the data you need for classifying new instances
    return 0


def classify_nb(classifier_data, document):
    # ...
    # return the prediction of the classifier
    return 0


def main():
    filepath = "./data/txt_sentoken"
    train, test = read_data(filepath)
    probabilities = train_nb(test)
    results = classify_nb(probabilities, test)


if __name__ == "__main__":
    main()
