import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

df = pd.read_csv('/data/clickbait_data.csv')
df.head(5)
df.shape

# train test split
text = df['headline'].values
labels = df['clickbait'].values
x_train, x_test, y_train, y_test = train_test_split(text, labels)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# intialisation
vocab_size = 5000
max_length = 500
embedding_size = 32

# tokenisation
tok = Tokenizer(num_words=vocab_size)
tok.fit_on_texts(text)
# transform text to sequence of integers
x_train_t = tok.texts_to_sequences(x_train)
x_test_t = tok.texts_to_sequences(x_test)
# pad sequences to make them of the same length
x_train_t = pad_sequences(x_train_t, maxlen=max_length)
x_test_t = pad_sequences(x_test_t, maxlen=max_length)

# define model
model = Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length=max_length))
model.add(LSTM(32, return_sequences=True))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()

callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        min_delta=1e-4,
        patience=3,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='weights.h5',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
]

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train_t, y_train,
                    batch_size=512,
                    validation_data=(x_test_t, y_test),
                    epochs=20,
                    callbacks=callbacks)

model.load_weights('weights.h5')
model.save('clickbait-model')

# plotting metrics
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
x = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x, acc, 'b', label='Training accuracy')
plt.plot(x, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(x, loss, 'b', label='Training loss')
plt.plot(x, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# confusion matrix
preds = [round(i[0]) for i in model.predict(x_test_t)]
cm = confusion_matrix(y_test, preds)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.xticks(range(2), ['Not clickbait', 'Clickbait'], fontsize=16)
plt.yticks(range(2), ['Not clickbait', 'Clickbait'], fontsize=16)
plt.show()

# precision and recall
tn, fp, fn, tp = cm.ravel()

precision = tp/(tp+fp)
recall = tp/(tp+fn)

print("Precision of the model is {:.2f}".format(recall))
print("Recall of the model is {:.2f}".format(precision))

# testing the model on random user input
test = ['How to Achieve Results Using This One Weird Trick', 'Learning data science from Coursera', 'A brief introduction to tensorflow', '12 things NOT to do as a Data Scientist']
token_text = pad_sequences(tok.texts_to_sequences(test), maxlen=max_length)
preds = [round(i[0]) for i in model.predict(token_text)]
for (text, pred) in zip(test, preds):
    label = 'Clickbait' if pred == 1.0 else 'Not Clickbait'
    print("{} - {}".format(text, label))
