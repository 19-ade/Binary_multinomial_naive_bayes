# Binary_multinomial_naive_bayes
Binary multinomial NB theorem applied from scratch for sentiment analysis . [This](https://view.datalore.jetbrains.com/notebook/8yGz8vdJuljg6JIhffWgxW) is the original datalore notebook where i made the project . I exported the .ipynb for this project.

## Naive Bayes Classification

This is a bayesian Classifier which makes a simplifying (naive) assumption about how the features interact.

We represent the sentences as a **bag_of_words**. A bag of words is basically an unordered set of words where the position does not matter but we keep track of the frequency of each word.

Now our main task is to find the class given a document . Here the class represents the sentiment. 

C<sub>nb</sub>=argmax<sub>c∈C</sub> P(c|d)

Here we are using Bayes Theorem to predict the class. Now according to Bayes theorem :

P(x|y)=P(y|x)P(x)/P(y)

applying this in the Naive Bias eqt:

C<sub>nb</sub>=argmax<sub>c∈C</sub>P(d|c)P(c)/P(d)

Now since the P(d) will remain constant throughout the process we can drop that from the equation.

so the equation turns into:

C<sub>nb</sub>=argmax<sub>c∈C</sub>P(d|c)P(c)

Here the P(c) is called the **prior probability** while P(d|c) is called the **likelihood probability**

Now to analyse the document we concentrate onto words (w<sub>i</sub>)

We assume two importanat things:
- The bag of words concept. The position of the word does not matter in the sentence.
- The probability of each factor(here words) is independent of each other 
so: P(w<sub>1</sub>,w<sub>2</sub>,.......|c)= P(w<sub>1</sub>|c)* P(w<sub>2</sub>|c)....

So finally we get:

C<sub>nb</sub>=argmax<sub>c∈C</sub>P(c)Π<sub>(i∈positions)</sub>   P(w<sub>i</sub>|c)

where P(wi/C)=count(wi,c)+a/x+V* a , where a is the laplace smoothing function, x is the length of the class c and V is the total vocabulary(total length of the bag_of_words)

Heres an example:












This was Multinomial Naive Bayes....


We are going to implement the Binary Multinomial Naive Bayes for sentiment analysis  which differs with Multinomial in the respect that in binary NB the repition of the words in each documents  are not counted. The presence of specific words matters more than the frequency .






For example:







## Resources:
- I used this [resource](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwir0szqodjuAhXJfX0KHWk3D6kQFjAQegQIIBAC&url=https%3A%2F%2Fweb.stanford.edu%2F~jurafsky%2Fslp3%2F4.pdf&usg=AOvVaw00iILqjBPC8ocD9_czKqu2) for learning about Naive Bayes . All the pictures used here are also from the same resource.
- The dataset used is included in the repo. I used the randomized version of the same for training and testing the NB classifier. The data was obtained from [here](https://github.com/chen0040/mxnet-sentiment-analysis/blob/master/demo/data/umich-sentiment-train.txt)




  
