import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from nltk.corpus import stopwords
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from tensorflow import keras

# |%%--%%| <unLOBrzI3U|ry2Sm5bleP>

EMBEDDING_DIM = 16
EPOCHS = 5
MAX_LEN = 32
MAX_WORDS = 20000

TEST_SIZE = 0.2
PROBABILITY_THRESHOLD = 0.5

# |%%--%%| <ry2Sm5bleP|Y8UJMinitd>

PATH_DATASET = "./kaggle/spam.csv"

# |%%--%%| <Y8UJMinitd|hoNDejUZKR>

dataset = pd.read_csv(
    PATH_DATASET,
    encoding="ISO-8859-1",
    usecols=[0, 1],
    skiprows=1,
    names=["label", "message"],
)
dataset.label = dataset.label.map({"ham": 0, "spam": 1})

# |%%--%%| <hoNDejUZKR|2RE4ItNaiZ>

x_train, x_test, y_train, y_test = train_test_split(
    dataset.message, dataset.label, test_size=TEST_SIZE
)

# |%%--%%| <2RE4ItNaiZ|zkQhRQd7a5>


# Tentando remover palavras comuns como "the", "a", "an", "in", etc
nltk.download("stopwords")
stop = set(stopwords.words("english"))


# https://stackoverflow.com/questions/5486337/how-to-remove-stop-words-using-nltk-or-python
def remove_stopwords(text: str):
    filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(filtered_words)


# |%%--%%| <zkQhRQd7a5|9jCxpDoKnh>


dataset["message"] = dataset.message.map(remove_stopwords)

# |%%--%%| <9jCxpDoKnh|99Cjm1uKXM>

# TODO: should we actually use num_unique_words as max_tokens? There's going to be more data adter the training
vectorizer = layers.TextVectorization(
    max_tokens=MAX_WORDS,
    output_mode="int",
    output_sequence_length=MAX_LEN,
    encoding="ISO-8859-1",
)

# |%%--%%| <99Cjm1uKXM|JPmT7xl9NJ>

vectorizer.adapt(x_train.to_numpy())

# |%%--%%| <JPmT7xl9NJ|AcofNVBCef>

model = keras.Sequential()
model.add(vectorizer)
model.add(layers.Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN))
model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
model.add(layers.Bidirectional(layers.LSTM(64)))
model.add(layers.Flatten())
model.add(layers.Dense(24, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

# |%%--%%| <AcofNVBCef|hl9TcorIIk>

model.summary()

# |%%--%%| <hl9TcorIIk|iFwZOu8fwt>

optimizer = keras.optimizers.Adam()
loss = keras.losses.BinaryCrossentropy()
metrics = keras.metrics.BinaryAccuracy()
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# |%%--%%| <iFwZOu8fwt|HsHN9kKDZu>

# Treinando o modelo

history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
)

# |%%--%%| <HsHN9kKDZu|yqaMtIpIFh>

# Perda e Acurácia
loss, accuracy = model.evaluate(x_test, y_test)
print("Perda:", loss)
print("Acurácia:", accuracy)

# |%%--%%| <yqaMtIpIFh|TIEPotKNVI>


# Fazendo predições para os dados de teste

predictions_prob = model.predict(x_test)
predictions = [1 if p > PROBABILITY_THRESHOLD else 0 for p in predictions_prob]

# |%%--%%| <TIEPotKNVI|odGJDavjN1>

# Criando a matrix de confusão para precisão e recall

confusion_matrix = tf.math.confusion_matrix(labels=y_test, predictions=predictions)
confusion_matrix

# |%%--%%| <odGJDavjN1|RWyVCze1kQ>

# Percentagem do total de spams detectados i.e., recall

recall = keras.metrics.Recall()
recall.update_state(y_test, predictions)
recall_score = recall.result().numpy()
print("Recall:", recall_score)

# |%%--%%| <RWyVCze1kQ|invCGMob7q>

# Percentagem das predições positivas (spams) corretas, i.e. precisão

precision = keras.metrics.Precision()
precision.update_state(y_test, predictions)
precision_score = precision.result().numpy()

print("Precisão:", precision_score)

# |%%--%%| <invCGMob7q|PTxwUG0zAg>

# Plotando a curva precisão-recall

precisions, recalls, _ = precision_recall_curve(y_test, predictions_prob)

plt.figure(figsize=(10, 7))
plt.plot(precisions[:-1], recalls[:-1])
plt.xlabel("Recalls")
plt.xticks(np.arange(0, 1.1, 0.1))
plt.ylabel("Precisão")
plt.yticks(np.arange(0, 1.1, 0.1))
plt.title("Curva Precisão-Recall")
plt.show()

# |%%--%%| <PTxwUG0zAg|nqOzfNZF4f>

# Exemplo de predição

sample_text = ["WINNER. You won this exciting lottery!!!"]
sample_input = pd.Series(sample_text)
sample_prob = model.predict(sample_input)
sample_pred = "SPAM" if sample_prob[0] > PROBABILITY_THRESHOLD else "HAM"
sample_pred

# |%%--%%| <nqOzfNZF4f|ZLPIvdhHc4>
