# A naive bayes sentiment classifier
import sklearn


def read_data(filepath):
    # read the data and transform it into a form that sklearn NB can use
    # use sklearn.datasets.load_files and sklearn.feature_extraction.text.CountVectorizer
    raw_data = sklearn.datasets.load_files(filepath, encoding="ASCII")
    vectorizer = sklearn.feature_extraction.text.CountVectorizer()
    X = vectorizer.fit_transform(raw_data)
    



def train_nb(training_documents):
    # ...
    # return the data you need for classifying new instances


def classify_nb(classifier_data, document):
    # ...
    # return the prediction of the classifier


def main():
    filepath = "./data/txt_sentoken"
    train, test = read_data(filepath)
    probabilities = train_nb(test)
    results = classify_nb(probabilities, test)


if __name__ == "__main__":
    main()
