import keras.models
import keras.layers


hashing_dimension = 1000


emb1 = keras.layers.Embedding(hashing_dimension, 64, input_length=10)











model = keras.models.Sequential()
