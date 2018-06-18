from konlpy.tag import Hannanum
import numpy as np
import keras.preprocessing.text

hashing_dimension = 1000

# bad programming practice lol
global_hannanum = Hannanum()

def tokenize(text):
    return list(p[0] + "/" + p[1] for p in global_hannanum.pos(text))


def read_data_file(data_filename):
    rows = []
    with open(data_filename, 'r') as file:
        rows = file.readlines()

    parsed = list(map(parse_row, rows))
    (X, y) = divide_to_xy(parsed)
    return (hash_data(X), hash_data(y))


# returns a tuple of form
# sbj entity
# obj entity
# relation
# original source sentence (ie. sbj and obj tags replaced) tokenized
def parse_row(data_row):
    parts = data_row.split("\t")
    obj = parts[0]
    sbj = parts[1]
    relation = parts[2]
    sentence = parts[3]

    sentence = sentence.replace("<< _sbj_ >>", sbj)
    sentence = sentence.replace("<< _obj_ >>", obj)

    return (obj, sbj, relation, tokenize(sentence))


def divide_to_xy(parsed_data):
    X = list(dp[3] for dp in parsed_data)
    y = list(dp[:3] for dp in parsed_data)

    return (X, y)


def hash_data(data):
    hash = lambda text: keras.preprocessing.text.hashing_trick(text, hashing_dimension)
    return list( list(map(hash, dp)) for dp in data)
