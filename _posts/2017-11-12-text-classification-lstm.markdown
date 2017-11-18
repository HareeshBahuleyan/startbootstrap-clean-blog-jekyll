---
layout:     post
title:      "Text Classification using LSTM Networks"
subtitle:   "Carry out sentiment analysis on the movie review dataset using a basic LSTM"
date:       2017-11-12 12:00:00
author:     "Hareesh Bahuleyan"
background: "/img/img-sentiment-analysis.jpg"
---

<link href="https://fonts.googleapis.com/css?family=Raleway:300" rel="stylesheet">

<style type="text/css">
	p {
	    font-size: 17px;
	    font-family: 'Raleway', sans-serif;
	    text-align: justify;
	}
	
	h2.subheading, li {
	    font-family: 'Raleway', sans-serif;
	}
</style>

It has been almost a year since I posted on my blog. I have been quite busy with my research work. In the past few months, I had the opportunity to gain some hands-on experience with deep learning. Looking at the number of papers that get published every week, there is no doubt that this is the right time to get into this field and work on some interesting problems. I have been working on some deep learning applications to NLP and on the way, I had the chance to familiarize myself with libraries such as <a href="https://keras.io">Keras</a> and <a href="https://www.tensorflow.org">Tensorflow</a>. Enough of where I have been all this while, lets dive into the tutorial!


## Sentiment Analysis
According to wikipedia, the aim of sentiment analysis is to determine the attitude of a speaker or writer towards a topic. Consumers usually tend to have opinions about products or services that they use, which they usually express through user reviews. Users can speak good about the product or may say that they were totally disappointed by the services offered. Now a product manager can go through all reviews and get an overall idea of how users feel about the product. Wouldn't it be much more convenient if your machine learning algorithm can automate this process? That is, it goes through all the reviews and gives you a final score for the product. One can do more complicated things like identifying certain aspects of the products that the users liked/disliked - this is called <a href="https://en.wikipedia.org/wiki/Sentiment_analysis#Feature.2Faspect-based">Aspect Based Sentiment Analysis</a>. 

## Long Short Term Memory Networks
Traditional neural networks are not suited to handle sequence data since they are unable to do reasoning based on previous events. Recurrent neural networks (RNN) tackle this issue by 'remembering' information from the previous time steps. The vanilla RNNs are rarely used because they suffer from the vangishing/exploding gradient problem. LSTMs or Long Short Term Memory Networks address this problem and are able to better handle 'long-term dependencies' by maintaining something called the cell state. The inflow and outflow of information to the cell state is contolled by three gating mechanisms, namely input gate, output gate and forget gate. I will not be going into the equations and mathematical details of LSTMs. If you are interested, the blog posts by <a href="http://colah.github.io/posts/2015-08-Understanding-LSTMs/">Chris Olah</a> and <a href="http://karpathy.github.io/2015/05/21/rnn-effectiveness/">Andrej Karpathy</a>.

## Hands-On
Moving on to the actual code, I have used Keras to implement this project. First, I shall implement a very simple network with just 3 layers - an Embedding Layer, an LSTM layer and an output layer with a sigmoid activation function. I later modify the network and provide some tips that can be useful in order to achieve a better performance. You can find the Jupyter notebook on <a href="https://github.com/HareeshBahuleyan/deeplearning-tutorials">my Github</a>. Here, I will be explaining only the important snippets of code.

### Keras
Keras is a very high level framework for implementing deep neural networks in Python. It is build on top on frameworks such as Tensorflow, Theano and CNTK. One can use any of the three as the backend while writing Keras code. Keras is convenient to build simple networks in the sense that it involves just writing code for blocks of the neural network and connecting them together from start to end.  

### Dataset
The Rotten Tomatoes movie review dataset originally released by Pang and Lee, is a benchmark dataset in the area of sentiment analysis. It consists of about 11,000 sentences, half of which have a positive label and the other half have a negative label. The dataset can be downloaded from <a href="http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz">this link</a>.

### Data Pre-processing
In this section, we load the data from text files and create a pandas dataframe. We also make use of pre-trained word embeddings. Word embeddings are one of the ways to represent words as vectors. These vectors are obtained in a way that captures distributional semantics of words, that is, words that have a similar meaning or occur in a similar context tend to have a similar representation and hence would be closer to each other in the vector space. You can read <a href="https://machinelearningmastery.com/what-are-word-embeddings/">this article</a> for a better understanding of word embeddings. GloVe and word2vec are the most popular word embeddings used in the literature. I will use 300d word2vec embeddings trained on the Google news corpus in this project, and it can be downloaded <a href="https://code.google.com/archive/p/word2vec/">here</a>. 

<div style="background: #f8f8f8; overflow:auto;width:auto;border:solid gray;border-width:.1em .1em .1em .8em;padding:.2em .6em;"><pre style="margin: 0; line-height: 125%"><span style="color: #008800; font-style: italic"># Read the text files of positive and negative sentences</span>
<span style="color: #AA22FF; font-weight: bold">with</span> <span style="color: #AA22FF">open</span>(DATA_DIR<span style="color: #666666">+</span><span style="color: #BB4444">&#39;rt-polarity.neg&#39;</span>, <span style="color: #BB4444">&#39;r&#39;</span>, errors<span style="color: #666666">=</span><span style="color: #BB4444">&#39;ignore&#39;</span>) <span style="color: #AA22FF; font-weight: bold">as</span> f:
    neg <span style="color: #666666">=</span> f<span style="color: #666666">.</span>readlines()
    
<span style="color: #AA22FF; font-weight: bold">with</span> <span style="color: #AA22FF">open</span>(DATA_DIR<span style="color: #666666">+</span><span style="color: #BB4444">&#39;rt-polarity.pos&#39;</span>, <span style="color: #BB4444">&#39;r&#39;</span>, errors<span style="color: #666666">=</span><span style="color: #BB4444">&#39;ignore&#39;</span>) <span style="color: #AA22FF; font-weight: bold">as</span> f:
    pos <span style="color: #666666">=</span> f<span style="color: #666666">.</span>readlines()

<span style="color: #AA22FF; font-weight: bold">print</span>(<span style="color: #BB4444">&#39;Number of negative sentences:&#39;</span>, <span style="color: #AA22FF">len</span>(neg))
<span style="color: #AA22FF; font-weight: bold">print</span>(<span style="color: #BB4444">&#39;Number of positive sentences:&#39;</span>, <span style="color: #AA22FF">len</span>(pos))

<span style="color: #008800; font-style: italic"># Create a dataframe to store the sentence and polarity as 2 columns</span>
df <span style="color: #666666">=</span> pd<span style="color: #666666">.</span>DataFrame(columns<span style="color: #666666">=</span>[<span style="color: #BB4444">&#39;sentence&#39;</span>, <span style="color: #BB4444">&#39;polarity&#39;</span>])
df[<span style="color: #BB4444">&#39;sentence&#39;</span>] <span style="color: #666666">=</span> neg <span style="color: #666666">+</span> pos
df[<span style="color: #BB4444">&#39;polarity&#39;</span>] <span style="color: #666666">=</span> [<span style="color: #666666">0</span>]<span style="color: #666666">*</span><span style="color: #AA22FF">len</span>(neg) <span style="color: #666666">+</span> [<span style="color: #666666">1</span>]<span style="color: #666666">*</span><span style="color: #AA22FF">len</span>(pos)
df <span style="color: #666666">=</span> df<span style="color: #666666">.</span>sample(frac<span style="color: #666666">=1</span>, random_state<span style="color: #666666">=10</span>) <span style="color: #008800; font-style: italic"># Shuffle the rows</span>
df<span style="color: #666666">.</span>reset_index(inplace<span style="color: #666666">=</span><span style="color: #AA22FF">True</span>, drop<span style="color: #666666">=</span><span style="color: #AA22FF">True</span>)
</pre></div>

The text data needs to be pre-processed into a sequence of numbers to be fed into a neural network. At this point, we need to assign an numerical index to each word in our corpus. But it may not always be possible to consider each and every word in our corpus. So we limit our vocabulary size to a maximum value (say 20,000), i.e., we consider only the most frequently occuring 20,000 words from the corpus. The remaining words can be replaced with a <UNK> token. In our case, as you would see in the code snippet below, the number of distinct words is below the vocabulary limit. 

Sentences in the dataset may be of varying lengths (number of words). However, we need to provide a fixed size input to the model. So the next decision that needs to be made is what sentence length would be ideal. Choosing a low value, say first 10 words of each sentence, we are able to reduce the computational compexity but at the cost of losing out some information (that might have been useful in classifying the polarity of the sentence). Analyzing the sentences, we find that 90 percentile of the sentences have atmost 31 words. In my case, I decide to pick 30 as the maximum sequence length. So sentences longer than 30 words with be truncated, and the shorter sentences will be zero padded. 

To illustrate the concepts described above, imagine that my maximum sequence limit is sequence length is set to 10. An I have the following sentence: <i>'The movie was totally WORTHLESS!'</i>. First, I would remove all puctuations and convert the words to lower case, then covert the sentence into a list of words. The words are then replaced by their corresponding indices by looking up a dictionary. The final step is to pad/truncate the sequence. Since the maximum sequence length is 10, the list is expanded to that size by filling in with zero values. 

<div style="background: #f8f8f8; overflow:auto;width:auto;border:solid gray;border-width:.1em .1em .1em .8em;padding:.2em .6em;"><pre style="margin: 0; line-height: 125%">Word Tokenize:  [<span style="color: #BB4444">&#39;the&#39;</span>, <span style="color: #BB4444">&#39;movie&#39;</span>, <span style="color: #BB4444">&#39;was&#39;</span>, <span style="color: #BB4444">&#39;totally&#39;</span>, <span style="color: #BB4444">&#39;worthless&#39;</span>]
Index Sequence: [<span style="color: #666666">1</span>, <span style="color: #666666">12</span>, <span style="color: #666666">4</span>, <span style="color: #666666">245</span>, <span style="color: #666666">358</span>]
Zero Padding:   [<span style="color: #666666">1</span>, <span style="color: #666666">12</span>, <span style="color: #666666">4</span>, <span style="color: #666666">245</span>, <span style="color: #666666">358</span>, <span style="color: #666666">0</span>, <span style="color: #666666">0</span>, <span style="color: #666666">0</span>, <span style="color: #666666">0</span>, <span style="color: #666666">0</span>]
</pre></div>

The way all of the above is done in keras is pretty simple:

<div style="background: #f8f8f8; overflow:auto;width:auto;border:solid gray;border-width:.1em .1em .1em .8em;padding:.2em .6em;"><pre style="margin: 0; line-height: 125%"><span style="color: #008800; font-style: italic"># Pre-processing involves removal of puctuations and converting text to lower case</span>
word_seq <span style="color: #666666">=</span> [text_to_word_sequence(sent) <span style="color: #AA22FF; font-weight: bold">for</span> sent <span style="color: #AA22FF; font-weight: bold">in</span> df[<span style="color: #BB4444">&#39;sentence&#39;</span>]]
<span style="color: #AA22FF; font-weight: bold">print</span>(<span style="color: #BB4444">&#39;90th Percentile Sentence Length:&#39;</span>, np<span style="color: #666666">.</span>percentile([<span style="color: #AA22FF">len</span>(seq) <span style="color: #AA22FF; font-weight: bold">for</span> seq <span style="color: #AA22FF; font-weight: bold">in</span> word_seq], <span style="color: #666666">90</span>)) <span style="color: #008800; font-style: italic"># Output: 31</span>

tokenizer <span style="color: #666666">=</span> Tokenizer(num_words<span style="color: #666666">=</span>MAX_VOCAB_SIZE)
tokenizer<span style="color: #666666">.</span>fit_on_texts([<span style="color: #BB4444">&#39; &#39;</span><span style="color: #666666">.</span>join(seq[:MAX_SENT_LEN]) <span style="color: #AA22FF; font-weight: bold">for</span> seq <span style="color: #AA22FF; font-weight: bold">in</span> word_seq])

<span style="color: #AA22FF; font-weight: bold">print</span>(<span style="color: #BB4444">&quot;Number of words in vocabulary:&quot;</span>, <span style="color: #AA22FF">len</span>(tokenizer<span style="color: #666666">.</span>word_index)) <span style="color: #008800; font-style: italic"># Output: 19180</span>

<span style="color: #008800; font-style: italic"># Convert the sequence of words to sequnce of indices</span>
X <span style="color: #666666">=</span> tokenizer<span style="color: #666666">.</span>texts_to_sequences([<span style="color: #BB4444">&#39; &#39;</span><span style="color: #666666">.</span>join(seq[:MAX_SENT_LEN]) <span style="color: #AA22FF; font-weight: bold">for</span> seq <span style="color: #AA22FF; font-weight: bold">in</span> word_seq])
X <span style="color: #666666">=</span> pad_sequences(X, maxlen<span style="color: #666666">=</span>MAX_SENT_LEN, padding<span style="color: #666666">=</span><span style="color: #BB4444">&#39;post&#39;</span>, truncating<span style="color: #666666">=</span><span style="color: #BB4444">&#39;post&#39;</span>)

y <span style="color: #666666">=</span> df[<span style="color: #BB4444">&#39;polarity&#39;</span>]

X_train, X_test, y_train, y_test <span style="color: #666666">=</span> train_test_split(X, y, random_state<span style="color: #666666">=10</span>, test_size<span style="color: #666666">=0.1</span>)
</pre></div>

The last line in the snippet above just splits the data into training and validation sets using the functionality from <code>sklearn</code>. Next we load the embeddings and create an embedding look up matrix as follows. With the help of this matrix, for any given word in our vocabulary, we would be able to lookup the 300d embedding vector or that word. In case, the word is not present in the pre-trained list of embeddings, then we use a randomly initialized vector of 300-dimension. 

<div style="background: #f8f8f8; overflow:auto;width:auto;border:solid gray;border-width:.1em .1em .1em .8em;padding:.2em .6em;"><pre style="margin: 0; line-height: 125%"><span style="color: #008800; font-style: italic"># Load the word2vec embeddings </span>
embeddings <span style="color: #666666">=</span> gensim<span style="color: #666666">.</span>models<span style="color: #666666">.</span>KeyedVectors<span style="color: #666666">.</span>load_word2vec_format(W2V_DIR, binary<span style="color: #666666">=</span><span style="color: #AA22FF">True</span>)

<span style="color: #008800; font-style: italic"># Create an embedding matrix containing only the word&#39;s in our vocabulary</span>
<span style="color: #008800; font-style: italic"># If the word does not have a pre-trained embedding, then randomly initialize the embedding</span>
embeddings_matrix <span style="color: #666666">=</span> np<span style="color: #666666">.</span>random<span style="color: #666666">.</span>uniform(<span style="color: #666666">-0.05</span>, <span style="color: #666666">0.05</span>, size<span style="color: #666666">=</span>(<span style="color: #AA22FF">len</span>(tokenizer<span style="color: #666666">.</span>word_index)<span style="color: #666666">+1</span>, EMBEDDING_DIM)) <span style="color: #008800; font-style: italic"># +1 is because the matrix indices start with 0</span>
<span style="color: #AA22FF; font-weight: bold">for</span> word, i <span style="color: #AA22FF; font-weight: bold">in</span> tokenizer<span style="color: #666666">.</span>word_index<span style="color: #666666">.</span>items(): <span style="color: #008800; font-style: italic"># i=0 is the embedding for the zero padding</span>
    <span style="color: #AA22FF; font-weight: bold">try</span>:
        embeddings_vector <span style="color: #666666">=</span> embeddings[word]
    <span style="color: #AA22FF; font-weight: bold">except</span> <span style="color: #D2413A; font-weight: bold">KeyError</span>:
        embeddings_vector <span style="color: #666666">=</span> <span style="color: #AA22FF">None</span>
    <span style="color: #AA22FF; font-weight: bold">if</span> embeddings_vector <span style="color: #AA22FF; font-weight: bold">is</span> <span style="color: #AA22FF; font-weight: bold">not</span> <span style="color: #AA22FF">None</span>:
        embeddings_matrix[i] <span style="color: #666666">=</span> embeddings_vector
        
<span style="color: #AA22FF; font-weight: bold">del</span> embeddings
</pre></div>

Finally, we have all the data ready and move on to the model building part. In this example, we use the <i>Sequential</i> Modelling API from Keras. In this framework, we basically define layers and stack them one over the other in a sequential manner. The other API is the <i>Functional</i> API, which we will keep it for some other day. 

<div style="background: #f8f8f8; overflow:auto;width:auto;border:solid gray;border-width:.1em .1em .1em .8em;padding:.2em .6em;"><pre style="margin: 0; line-height: 125%"><span style="color: #008800; font-style: italic"># Build a sequential model by stacking neural net units </span>
model <span style="color: #666666">=</span> Sequential()
model<span style="color: #666666">.</span>add(Embedding(input_dim<span style="color: #666666">=</span><span style="color: #AA22FF">len</span>(tokenizer<span style="color: #666666">.</span>word_index)<span style="color: #666666">+1</span>,
                          output_dim<span style="color: #666666">=</span>EMBEDDING_DIM,
                          weights <span style="color: #666666">=</span> [embeddings_matrix], trainable<span style="color: #666666">=</span><span style="color: #AA22FF">False</span>, name<span style="color: #666666">=</span><span style="color: #BB4444">&#39;word_embedding_layer&#39;</span>, 
                          mask_zero<span style="color: #666666">=</span><span style="color: #AA22FF">True</span>))

model<span style="color: #666666">.</span>add(LSTM(LSTM_DIM, return_sequences<span style="color: #666666">=</span><span style="color: #AA22FF">False</span>, name<span style="color: #666666">=</span><span style="color: #BB4444">&#39;lstm_layer&#39;</span>))

model<span style="color: #666666">.</span>add(Dense(<span style="color: #666666">1</span>, activation<span style="color: #666666">=</span><span style="color: #BB4444">&#39;sigmoid&#39;</span>, name<span style="color: #666666">=</span><span style="color: #BB4444">&#39;output_layer&#39;</span>))
</pre></div>

The first layer is an Embedding layer, which takes as input a sequence of integer indices and returns a matrix of word embeddings. The next layer is an LSTM which processes the sequence of word vectors. Here, the output of the LSTM network is 128-dimensional vector which is fed into a dense network with a sigmoid activation in order to output a probability value. This probability can then be rounded off to get the predicted class label, 0 or 1. The <code>model.summary()</code> function is useful to verify the model in terms of what goes as inputs and outputs and what are the shaped of those. We also get an idea of the number of parameters that the model has to train and optimize. In this case, the non-trainable parameters are the word-embeddings.

<div style="background: #f8f8f8; overflow:auto;width:auto;border:solid gray;border-width:.1em .1em .1em .8em;padding:.2em .6em;"><pre style="margin: 0; line-height: 125%">_________________________________________________________________
Layer (<span style="color: #AA22FF">type</span>)                 Output Shape              Param    
<span style="color: #666666">=================================================================</span>
word_embedding_layer (Embedd (<span style="color: #AA22FF">None</span>, <span style="color: #AA22FF">None</span>, <span style="color: #666666">300</span>)         <span style="color: #666666">5754300</span>   
_________________________________________________________________
lstm_layer (LSTM)            (<span style="color: #AA22FF">None</span>, <span style="color: #666666">128</span>)               <span style="color: #666666">219648</span>    
_________________________________________________________________
output_layer (Dense)         (<span style="color: #AA22FF">None</span>, <span style="color: #666666">1</span>)                 <span style="color: #666666">129</span>       
<span style="color: #666666">=================================================================</span>
Total params: <span style="color: #666666">5</span>,<span style="color: #666666">974</span>,<span style="color: #666666">077</span>
Trainable params: <span style="color: #666666">219</span>,<span style="color: #666666">777</span>
Non<span style="color: #666666">-</span>trainable params: <span style="color: #666666">5</span>,<span style="color: #666666">754</span>,<span style="color: #666666">300</span>
_________________________________________________________________
</pre></div>


One can also get a visual feel of the model by using the <code>plot_model</code> utility in Keras. 

<center>
<a href="#">
    <img src="{{ site.baseurl }}/img/Post-6-LSTM-Text-Classification/basic_lstm_classifier.png" alt="Model Plot">
</a>
</center>

The last steps are pretty simple. The model needs to be compiled before actually training. Here, we need to specify the loss function, the optimizer and the metrics that needs to be monitored throughout the training. The training process will begin once the <code>model.fit()</code> is called. The arguments to this function include the number of epochs of training, batch size and the training and validation sets. One epoch refers to one run through the entire training dataset. The training data is split into batches and the model parameters are updated after each batch is processed. 

<div style="background: #f8f8f8; overflow:auto;width:auto;border:solid gray;border-width:.1em .1em .1em .8em;padding:.2em .6em;"><pre style="margin: 0; line-height: 125%">model<span style="color: #666666">.</span>compile(loss<span style="color: #666666">=</span><span style="color: #BB4444">&#39;binary_crossentropy&#39;</span>,
              optimizer<span style="color: #666666">=</span><span style="color: #BB4444">&#39;adam&#39;</span>,
              metrics<span style="color: #666666">=</span>[<span style="color: #BB4444">&#39;accuracy&#39;</span>])

model<span style="color: #666666">.</span>fit(X_train, y_train,
          batch_size<span style="color: #666666">=</span>BATCH_SIZE,
          epochs<span style="color: #666666">=</span>N_EPOCHS,
          validation_data<span style="color: #666666">=</span>(X_test, y_test))
</pre></div>

<div style="background: #f8f8f8; overflow:auto;width:auto;border:solid gray;border-width:.1em .1em .1em .8em;padding:.2em .6em;"><pre style="margin: 0; line-height: 125%">score, acc <span style="color: #666666">=</span> model<span style="color: #666666">.</span>evaluate(X_test, y_test,
                            batch_size<span style="color: #666666">=</span>BATCH_SIZE)

<span style="color: #AA22FF; font-weight: bold">print</span>(<span style="color: #BB4444">&quot;Accuracy on Test Set = {0:4.3f}&quot;</span><span style="color: #666666">.</span>format(acc))
</pre></div>

On evaluating the model, I get an accuracy of 80.4%, which is decent considering the fact the the network that I used is pretty rudimentary. 

Next, lets try modifying the network above with the following tricks, in the hope of achieving a better performance:

- **<u>Bi-directional LSTM:</u>** The LSTM that I used reads the sequence in the forward direction, i.e., from the first word to the last word. We can try reading the sentence in a reverse fashion as well (which has been proven to do well in tasks such as POS tagging). In the bidirectional LSTM, the final hidden states of the foward and the backward LSTMs are concatenated and passed on to the downstream network.  

- **<u>Batch-normalization:</u>** This is a method introduced by <a href="https://arxiv.org/pdf/1502.03167.pdf">Ioffe & Szegedy</a>, that helps speed up the training process and simultaneously reduce over-fitting. To prevent instability of the network during training, we usually normalize the inputs to the model by subtract the mean value and dividing by the standard deviation. With batch-normalization, the output of a previous layer is normalized in the same manner, before it is fed into the next layer. For feed-forward networks, batch-normalization is carried out before applying the activation function.  

- **<u>Dropout:</u>** A simple yet powerful regularization technique, that states that some nodes in the network can be ignored during some iterations of training. That is, a given node will participate in the forward and backward pass (weight update), with the desired keep probability (which is <code>1.0 - dropout_probability</code>). The idea is that during training, the network learns to predict using <u>only a subset of the weights</u>. This would intuitively mean that during test time, when it can use <u>the entire set of weights</u>, it should have a better predicting power. 

- **<u>Trainable Embeddings:</u>** In the basic model, the pre-trained word embeddings were used as is. However, we have an option to allow even the word embeddings to be trainable parameters, i.e., they embeddings are adjusted during the gradient descent procedure. One needs to be aware that setting <code>trainable=True</code> can result in the word embeddings tuned to overfit on the training data and perform poorly on the validation set.

The code has been posted on <a href="https://github.com/HareeshBahuleyan/deeplearning-tutorials">my Github</a>. The accuracy increased to 81.5%, the performance improvement was not as much as I expected inspite of a more complicated network. If you are interested, you can play around with some of these parameters and see its effect on validation accuracy. 

This was a simple post that demonstrated the use of LSTMs for text classification. Sentiment analysis has been a widely explored problem - if you go beyond the usual sentences, you would need to handle issues such as negation and sarcasm in sentences. You could formulate this as a regression problem and obtain a score rather than outputting a class label. As for LSTMs, they are powerful and have been utilized in a wide variety of NLP tasks. I will be back with more deep learning stuff in my future blog posts. 

