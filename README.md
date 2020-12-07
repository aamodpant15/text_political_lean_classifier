# text_political_lean_classifier
Enter text as a string, and get a prediction for whether it is left leaning or right leaning.

## Disclaimer

Even though this project is based on politics, the crux of it, is consumption of biased information on the internet. I do not intend on making any political statements, or take sides. This is purely for internet users to be more aware, and potentially disregard information that could be highly biased.  
More and more nowadays, it is getting difficult to differentiate opinion pieces from legitimate factual news pieces online. In fact, many highly biased posts are written specifically to sound like legitimate news. It helps to do a quick check on maybe another piece of text by the author, or on that website, to be aware of their biases. This will help us be more informed on what we read on the internet.  
Having a bias does not necessarily mean that the content is wrong, but it is fair enough to have an idea of a particular bias, so all information can be considered with the legitimacy that it warrants.  
Equal number of posts were taken from left and right leaning subreddits to train the model.


## Description
This is a **Naive Bayes** model that trains on Bag of Words. Achieves **86% accuracy**.  
It is trained on the `data.xlsx` file, which consists of posts, and comments from over 30 politically aligned subreddits of both sides. This is a part of a larger project I did with some colleagues at University of Massachusetts, Amherst. We wrote a paper on this issue, and tested with two other models. This one was one of the better performing models.
