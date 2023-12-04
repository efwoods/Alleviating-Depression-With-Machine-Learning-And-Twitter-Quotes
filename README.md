<a href="https://gitpod.io/#https://github.com/efwoods/EvanWoodsTwitter">
  <img
    src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod"
    alt="Gitpod"
  />
</a>

#  Alleviating Depression With  Machine Learning And Twitter Quotes


 ### Purpose 
 I post tweets of my favorite inspiring quotes to people who are #depressed. I want to give them a moment where they feel connected to someone who cares. I hope they feel a moment of escape from their depression. Because this is my personal twitter account, I have chosen to share quotes that I find valuable. These quotes are a reflection of myself and my personal values, and I hope that sharing appropriately the wisdom of a quote that has aided me in my life will in turn, through the correct response to the right person in need, shine light into their dark world - if only for a moment.
 
  ### How Does It Work?
  `python tweet.py` will pull my tweets from my twitter, process them, and create a Bigram Language Model to generate tweets that are based off of tweets I would tweet. I then use autocorrect to post process the generated tweets. Next, I find an arbitrary number of tweets under #depressed. these tweets are preprocessed and analyzed for sentiment through a logistic regression. Negative tweets are used further in the pipeline, and positive sentiment tweets are ignored. The recent likes of each user of the #depressed tweets that have negative sentiment is collected, processed, and classified by a stochastic gradient descent (SGD) classifier as to which class the liked tweet belongs. There are 10 possible classes:
  - death
  - happiness
  - inspiration
  - love
  - poetry
  - romance
  - science
  - success
  - time
  - truth
  
  The mode of the classes of the liked tweets is taken to identify which class topic the user most likes. Next, tweets generated by my transformer are classified and split into subset groups based upon the classes above. The original depressed tweet is then included into the matching subset, and a cosine similiarity matrix is implemented to identify the most relevant generated tweet related to the depressed tweet. 
  
  This generated is one that is:
  - Relevant to the tweet the user originally sent.
  - Is of the class of topic that the user likes.
  - Concatonated with a *give hug* expression to emphasize the intent of the reply.

  The quote is then replied to and each tweet id is recorded so as not to duplicate responses when this process is repeated.  

  Because the Bigram Language Model is trained using my history of tweets, tweets that I post manually will update the model. Tweets that I edit, will also update the model of generated tweets. The more I use twitter, the closer to my voice the generated tweets will be. It is limited by the total number of tweets I can pull at a time. This indicates a 1000 tweet rolling average of my personality.

  I am developing methods to save states of my personality throughout this process.

 ### Models 
 This code base implements the following:
 - [A Logistic Regression](code/models/Sentiment-LR.pickle)
   - This model is used to make prediction of the positive or negative sentiment of the #depressed tweet.
 - [A Vectorizer](code/models/vectoriser-ngram-(1%2C2).pickle)
   - This model is used to convert the processed text into a tf-idf vector so the logistic regression can make a prediction.
 - [A Stochastic Gradient Descent Classifier](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
   - This model is used to predict which of 10 topics the depressed person most likes in their recent tweet history on twitter.
 - [A Cosine Similariy Matrix](https://en.wikipedia.org/wiki/Cosine_similarity)
   - This matrix will identify and recommend the most relevant quote to reply with on twitter in order to share a quote that is liked by both myself and the user. 
 - [Bigram Language Model](https://www.youtube.com/watch?v=kCc8FmEb1nY)
   - This is a from-scratch general pretrained transformer. It is used to generate tweets that are based on my history of tweets sent from my twitter account.

### Analysis
The graph below shows tweets replied to and tweets liked over time. Prior to January, the tweets were manually sent by hand. The logistic regression, vectorizer, and cosine similarity matrix were implemented in January. This increased the number of likes based upon sentiment. Not yet shown in the graph is the effect of the SGD classifier, which increased the number of likes from approximately 3 to 20 in a day. This graph will be ploted in the future.   

There is an outlier of a tweet that is liked just before November. This was a tweet that I replied to which had a large following. It was about the coverage of starlink satellites in Japan and is unrelated to this effort. 
## ![](analysis/Tweet_Replies_Sent_VS_Liked.png)

### Future Goals
- Plot the effect of the generated Tweets on likes
- Create generative models for different classes such as death, happiness, inspiration, love, poetry, romance, science, success, time, & truth
- Plot the effect of the SGD Classifier on the replied/liked ratio of tweets.

- Use a LLM such as Bloom to generate responses.
- Incorporate an ensemble of virtues that I embody to respond in ways that match my virtues when appropriate.
- Reply using a larger corpus, such as the [Goodreads-books dataset on kaggle](https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks) to send quotes that the user may enjoy that I may not but are none-the-less appropriate to be sent by myself so as to ease their depression for a moment.  

### New Features
- Incorporate tweet translation to reach out to non-english speaking users of twitter and tweet replies in their native tongue.
