### Data description

There just two columns in this dataset. The `headline` column which contains the text of the article headline and the `clickbait` column which contains the label `1` or `0` indicating whether or not the article is a clickbait.

### Train-Test split

We'll stick to the default values of `train_test_split`, which will keep the test size to 25% of the dataset. 

### Tokenisation

This is the process of splitting the text (phrase, sentence, paragraph) into smaller units such as individual terms. The importance of tokenisation is that it helps in understanding the context and aides the development of the model.

Let's discuss the three functions we've used in this section

1. `fit_on_texts`: This function creates the  vocabulary index based on word frequency. ... So it basically takes each word in the text and replaces it with its corresponding integer value  from the word_index dictionary.
2. `texts_to_sequences`:Transforms each text in texts to a sequence of integers. Only top `num_words - 1` most  frequent words will be taken into account. Only words known by the  tokenizer will be taken into account.
3. `pad_sequences`: This used to ensure that all sequences in a list have the same length. By default this is done by  padding 0 in the beginning of each sequence until each sequence has the  same length as the longest sequence. It will also truncate all sequnces longer than `max_length`

### Defining the model

We'll be using an an LSTM network. Below are each of the layers used:

1. Embedding: We use an embedding layer to compress the input feature space into a smaller one. magine that we have 80,000 unique words in a text classification problem and we select to preprocess the text and create a term document matrix. This matrix will be sparse and a sequence of the sequence  ['how',  'are', 'you'] is a 80,000-dimensional vector that is all zeros except  from 3 elements that correspond to those words. In the case, we pass  this matrix as input to the model it will need to calculate the weights  of each individual feature (80,000 in total). This approach is memory  intensive. 

   In short, it turns positive integers (indices) into dense vectors of a fixed size.

2. LSTM: This is an RNN architecture used in classifying tasks. Usually used in time series or any data which can have a lag since lags of unknown duration between two events can be very important. It is capable of learning order dependence in sequence prediction problems.

3. GlobalMaxPooling1D: This layer downsamples the input representation by taking the maximum value over the time dimension. 

4. Dropout: This layer randomly sets input units to 0 with a frequency of some *rate* at each step during training time, which helps prevent overfitting.

5. Dense: This layer feeds all outputs from the previous **layer** to all its neurons, each neuron providing one output to the next **layer**. It's the most basic **layer** in neural networks.

The model summary looks like:

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 500, 32)           160000    
_________________________________________________________________
lstm (LSTM)                  (None, 500, 32)           8320      
_________________________________________________________________
global_max_pooling1d (Global (None, 32)                0         
_________________________________________________________________
dropout (Dropout)            (None, 32)                0         
_________________________________________________________________
dense (Dense)                (None, 1)                 33        
=================================================================
Total params: 168,353
Trainable params: 168,353
Non-trainable params: 0
_________________________________________________________________
```

### Compiling the model

Here some key elements must be noted. As our loss metric, we're are using `binary_crossentropy`, which is used for classification algorithms and since ours is a two-class classification problem the keyword here is *binary*. Our optimiser of choice is `adam`, which is usually considered as an alternative to stochastic gradient descent. We choose `adam` because it implicitly performs coordinate-wise gradient clipping and can hence, unlike *SGD*, tackles heavy-tailed noise.

### Metrics

Below are two graphs for **Training and validation Accuracy** and **Training and validation Loss**, which show us that there is negligible overfitting. These are followed by the confusion matrix and recall and precision of the model. 

<p align="center">
  <img title='Training and Validation acc/loss' src='https://github.com/altprime/clickbait-classification/blob/main/output/1-clickbait-train-val-acc-loss.jpg'>
</p><br>
<p align="center">
  <img title='Confusion Matrix' src='https://github.com/altprime/clickbait-classification/blob/main/output/2-clickbait-confusion-matrix.jpg'>
</p>

```
Recall of the model is 0.97
Precision of the model is 0.98
```

### Random user input

Let's also have a look at some random user input and see for ourselves whether the model is performing as expected:

```
How to Achieve Results Using This One Weird Trick - Clickbait
Learning data science from Coursera - Not Clickbait
A brief introduction to tensorflow - Not Clickbait
12 things NOT to do as a Data Scientist - Clickbait
```

Seems like it's doing a pretty good job!

### Future scope

We could deveop a slightly more complex neural network on a much larger dataset and create a front end or a Chrome extension that displays `Clickbait` and `Not Clickbait` next to the articles on the search page.

