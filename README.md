# Sentiment Analysis

  Scratch approach:

  Aim of the model is to find whether a tweet referred as a postive sentiment or negative sentiment using Navie Bayes
  The naive bayes classfier is one of the simple probabilistic classifier used for processing text. we implemeted two methodds for naive bayes from scratch.

1.sentiment_analysis (slightly modified approach)
2.naive_bayes a standard approach
   
# 1.sentiment analysis(slightly modified approach
   pre-Prcoessing
    we have done pre processing only for the trianing data set 
    we have removed numbers from the data set, extra special characters.  we also trimmed exrtra white spaces. since it is not any known languages we cannot able reduce the stop word. we also tried clipping but ti deos decrease the accuracy so we avoided.
    
   Fitting the model
   i will find the total number of uniques tokens in the whole coropus 
   we first segeregate postive tweets and negative tweets, and then i will find the tolal number of words in the positive corpus and negative corpus. i will throw all the psotive words in one bag and negative words in bad along with their frequency.
   
   accuracy on the test_set: 0.5991
 
  To predict
    i will loop through all the words in the sentence and i will find two score for every word and multiply all the indicidual probability
    A the end of the predict fucntion i will be left with two socre for a single tweet.
    each tweet will have probability of sentence to be positive and porbability of sentence to be nagative . i will find the maximum of two and will predict.
    
# 2.Navie bayes standard approach
 we have done pre processing only for the trianing data set 
    we have removed numbers from the data set, extra special characters.  we also trimmed exrtra white spaces. since it is not any known languages we cannot able reduce the stop word. we also tried clipping but ti deos decrease the accuracy so we avoided.
    
   Fitting the model
   i will find the total number of uniques tokens in the whole coropus 
   we first segeregate postive tweets and negative tweets, and then i will find the tolal number of words in the positive corpus and negative corpus. i will throw all the psotive words in one bag and negative words in bad along with their frequency
   logprior = np.log(d_pos) - np.log(d_neg)
   Both are almost the ssame the only differece is , here at the end i will predict if my predicted score is greate than logprior value i will conclude it as psotive or else negative,  
   In the standard approach we used log probabilities to avoid underflow values.
   
   accuracy on the test_set: 0.781
   
# Anything Goes Approach

For the anything goes approach, we implemented the sentiment analysis using pre defined machine learning and deep learning library algorithms. In our method, we have compared Naive Bayes, Logistic Regression and Neural Network (LSTM).

It is observed that the accuracy of Naive Bayes is greater than the LSTM.

Data Preprocessing:

Before giving the text data as an input to these models, data pre processing is always done to ensure the models can run efficiently without any errors. For this, the text data is cleaned initially. Things like punctuations, whitespaces, emoji's, numbers are all removed from the dataset. It is then stemmed and lemmatized. After this, the dataset is vectorised in order to represent it in the form of a vector and is then tokenized so it is converted into a format that can be given as an input to the model. For this vectorisation method, we used n_grams in the range of (1,2) which is known as the standard range for vectorization. There are also many types of vectorisation algorithms. The one used in this approach is the CountVectorizer from Sklearn. This tokenized format of the sequence is then padded for allowing it to be of the same length which is further split into the training and testing sets. This is then given as the input to the models. 
   
 Logistic Regression:
 
Logistic Regression is a linear model that estimates the probability of an event occurring, such as voted or didn't vote, based on a given dataset of independent variables. Since the outcome is a probability, the dependent variable is bounded between 0 and 1. It is a discriminative classifier. For this model, the accuracy was 0.5125625.

Naive Bayes:

Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features given the value of the class variable. There are different types of Naive Bayes algorithm but the one that we have used here is the Gaussian NB. The accuracy for this model is approximately 0.82 which is really way better than the other two models trained.

LSTM:

Long Short-Term Memory (LSTM) networks are a type of recurrent neural network capable of learning order dependence in sequence prediction problems.  LSTM has feedback connections. Such a recurrent neural network (RNN) can process not only single data points (such as images), but also entire sequences of data (such as speech or video). For LSTM, we used sigmoid as the activation function, binary_crossentropy as the loss function, adam as the optimizer and accuracy score for the measuring metric. LSTM gave an accuracy of 0.5020 on the training dataset and 0.5200 accuracy for the testing dataset. 

Overall, it is observed that different models work on their own criteria depending upon the use cases like data size, data type and the goal of the analysis. We did find a lot of issue in the accuracy. We expected it to be more but I think there are certain reasons why the accuracy was around 50%. Maybe, in the future, we might have to tweek and tune the model more so that it's efficiency is more, resulting in higher accuracy. 


reference:https://medium.com/nerd-for-tech/illustrated-naive-bayes-implementation-from-scratch-for-sentiment-analysis-63c4bcab6053
