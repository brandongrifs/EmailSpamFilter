
# coding: utf-8

# Before you turn this assignment in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All). Lastly, hit **Validate**.
# 
# If you worked locally, and then uploaded your work to the hub, make sure to follow these steps:
# - open your uploaded notebook **on the hub**
# - hit the validate button right above this cell, from inside the notebook
# 
# These  steps should solve any issue related to submitting the notebook on the hub.
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[154]:


NAME = "Brandon Griffin"
COLLABORATORS = ""


# ---

# # Project 2: Spam // Ham Prediction  
# 
# ## Due Date: 11:59pm Sunday, April 29
# 
# In this project, you will use what you've learned in class to create a classifier that can distinguish spam (junk or commercial or bulk) emails from ham (non-spam) emails. In addition to providing some skeleton code to fill in, we will evaluate your work based on your model's accuracy and your written responses in this notebook.
# 
# ## Score breakdown
# 
# Question | Points
# --- | ---
# Question 1 | 3
# Question 2 | 2
# Question 3a | 2
# Question 3b | 2
# Question 4 | 2
# Question 5 | 2
# Question 6 | 9
# Question 7 | 6
# Question 8 | 6
# Question 9 | 3
# Question 10 | 5
# Total | 42

# # Part I - Initial Analysis

# In[155]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set(style = "whitegrid", 
        color_codes = True,
        font_scale = 1.5)


# ### Loading in the Data
# 
# The dataset consists of email messages and their labels (0 for ham, 1 for spam). Your labelled dataset contains 8348 labelled examples, and the evaluation set contains 1000 unlabelled examples.
# 
# Run the following cells to load in the data into DataFrames.
# 
# The `train` DataFrame contains labelled data that you will use to train your model. It contains three columns:
# 
# 1. `id`: An identifier for the training example.
# 1. `subject`: The subject of the email
# 1. `email`: The text of the email.
# 1. `spam`: 1 if the email was spam, 0 if the email was ham (not spam).
# 
# The `evaluation` DataFrame contains another set of 1000 unlabelled examples. You will predict labels for these examples and submit your predictions to Kaggle for evaluation.

# In[156]:


from utils import fetch_and_cache_gdrive
fetch_and_cache_gdrive('1SCASpLZFKCp2zek-toR3xeKX3DZnBSyp', 'train.csv')
fetch_and_cache_gdrive('1ZDFo9OTF96B5GP2Nzn8P8-AL7CTQXmC0', 'eval.csv')

original_training_data = pd.read_csv('data/train.csv')
evaluation = pd.read_csv('data/eval.csv')

# Convert the emails to lower case as a first step to processing the text
original_training_data['email'] = original_training_data['email'].str.lower()
evaluation['email'] = evaluation['email'].str.lower()

original_training_data.head()


# ## Train-Test Split
# 
# The training data we downloaded is all the data we have available for both training models and **testing** the models that we train.  We therefore need to split the training data into separate training and test datsets.  You will need this **test data** to evaluate your model once you are finished training.

# In[157]:


from sklearn.model_selection import train_test_split

[train, test] = train_test_split(original_training_data, test_size=0.1, random_state=42)


# # Question 1
# 
# In the cell below, print the text of the first ham and the first spam email in the training set. Then, discuss one thing you notice that is different between the two that might relate to the identification of spam.

# In[158]:


# Print the text of the first ham and the first spam emails. Then, fill in your response in the q01 variable:
first_ham = train[train['spam']==0]
first_spam = train[train['spam']==1]

print(first_ham['email'].head(1))
print(first_spam['email'].head(1))


# In[159]:


# This is a cell with just a comment but don't delete me if you want to get credit.


# The spam email has long chains of non-digit/letter characters and few appropriate spacings of such characters. There are letters and numbers next to each other more often than normal, and long chains of letters which likely could not correspond to real words. Underscores excessively may also be an indicator.

# # Basic Feature Engineering
# 
# We would like to take the text of an email and predict whether the text is ham or spam. This is a *classification* problem, so we can use logistic regression to make a classifier. Recall that to train an logistic regression model we need a numeric feature matrix $\Phi$ (pronounced phi as in wifi) and corresponding binary labels $Y$.  Unfortunately, our data are text, not numbers. To address this, we can create numeric features derived from the email text and use those features for logistic regression.
# 
# Each row of $\Phi$ is derived from one email example. Each column of $\Phi$  is one feature. We'll guide you through creating a simple feature, and you'll create more interesting ones when you are trying to increase your accuracy.

# # Question 2
# 
# Create a function called `words_in_texts` that takes in a list of `words` and a pandas Series of email `texts`. It should output a 2-dimensional NumPy array containing one row for each email text. The row should contain either a 0 or a 1 for each word in the list: 0 if the word doesn't appear in the text and 1 if the word does. For example:
# 
# ```python
# >>> words_in_texts(['hello', 'bye', 'world'], 
#                    pd.Series(['hello', 'hello world hello']))
# 
# array([[1, 0, 0],
#        [1, 0, 1]])
# ```

# In[160]:


def words_in_texts(words, texts):
    '''
    Args:
        words (list-like): words to find
        texts (Series): strings to search in
    
    Returns:
        NumPy array of 0s and 1s with shape (n, p) where n is the
        number of texts and p is the number of words.
    '''
    indicator_array =  np.zeros((len(texts),len(words)))
    count = 0
    for i in texts.index:
        for j in range(len(words)):
            indicator_array[count][j] = (words[j] in texts.loc[i])
        count = count + 1
    
    return indicator_array


# In[161]:


# If this doesn't error, your function outputs the correct output for this example
assert np.allclose(words_in_texts(['hello', 'bye', 'world'], 
                                  pd.Series(['hello', 'hello world hello'])),
                   np.array([[1, 0, 0], 
                             [1, 0, 1]]))


# # Basic EDA
# 
# Now we need to identify some features that allow us to tell spam and ham emails apart. One idea is to compare the distribution of a single feature in spam emails to the distribution of the same feature in ham emails. If the feature is itself a binary indicator, such as whether a certain word occurs in the text, this amounts to comparing the proportion of spam emails with the word to the proportion of ham emails with the word.
# 

# # Question 3a
# 
# Create a bar chart comparing the proportion of spam and ham emails containing certain words. It should look like the following plot (which was created using `sns.barplot`), but you should choose your own words as candidate features.
# 
# ![training conditional proportions](training_conditional_proportions.png "Class Conditional Proportions")
# 

# In[162]:


ham = train[train['spam']==0]['email']

spam = train[train['spam']==1]['email']

words = ['body', 'business', 'html', 'money', 'offer', 'please']
hamscanner = words_in_texts(words, train['email'])
count = len(hamscanner)
hamlist = pd.DataFrame(data=hamscanner, columns = words)
print(hamscanner)

hamlist['Title'] = train['spam'].values
hamlist['Title'].replace({0:'Ham', 1:'Spam'}, inplace=True)
hamlist = hamlist.melt('Title').groupby(['Title', 'variable']).mean().reset_index().rename(columns = {'variable': '', 'value': 'Proportion of Emails'})
#for p in hamscanner
sns.barplot(x='', y='Proportion of Emails', hue="Title", data=hamlist)
plt.ylim([0,1])


# # Question 3b
# 
# When the feature is binary, it makes sense (as in the previous question) to compare the proportion of 1s in the two classes of email. Otherwise, if the feature can take on many values, it makes sense to compare the distribution under spam to the distribution under ham. Create a *class conditional density plot* like the one below (which was created using `sns.distplot`), comparing the distribution of a feature among all spam emails to the distribution of the same feature among all ham emails. **You may use the Fraction of Uppercase Letters or create your own feature.**
# 
# ![training conditional densities](training_conditional_densities2.png "Class Conditional Densities")

# In[163]:


training_data2 = pd.read_csv('data/train.csv')
[train2, test2] = train_test_split(training_data2, test_size=0.1, random_state=42)


# In[164]:


#print(sum(1 for c in train2[train2['spam']==1]['email'].values if c.isupper()))
train2['percentage'] = train2['email'].str.findall('[A-Z]').str.len()/train2['email'].str.findall('[a-zA-Z]').str.len()


sns.distplot(train2[train2['spam']==0]['percentage'], label='Ham')
sns.distplot(train2[train2['spam']==1]['percentage'], label='Spam', axlabel='Fraction of Uppercase Letters in Email')
plt.legend()


# # Basic Classification
# 
# Notice that the output of `words_in_texts(words, train['email'])` is a numeric matrix containing features for each email. This means we can use it directly to train a classifier!

# # Question 4
# 
# We've given you 5 words that might be useful as features to distinguish spam/ham emails. Use these words as well as the `train` DataFrame to create two NumPy arrays: `Phi_train` and `Y_train`.
# 
# `Phi_train` should be a matrix of 0s and 1s created by using your `words_in_texts` function on all the emails in the training set.
# 
# `Y_train` should be a vector of the correct labels for each email in the training set.

# In[165]:


some_words = ['drug', 'bank', 'prescription', 'memo', 'private']

Phi_train = words_in_texts(some_words, train['email'])
Y_train = train['spam']



Phi_train[:5], Y_train[:5]


# In[166]:


assert np.all(np.unique(Phi_train) == np.array([0, 1]))
assert np.all(np.unique(Y_train) == np.array([0, 1]))
assert Phi_train.shape[0] == Y_train.shape[0]
assert Phi_train.shape[1] == len(some_words)


# # Question 5
# 
# Now we have matrices we can give to scikit-learn! Using the [`LogisticRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) classifier, train a logistic regression model using `Phi_train` and `Y_train`. Then, output the accuracy of the model (on the training data) in the cell below. You should get an accuracy of around 0.75.

# In[167]:


from sklearn import linear_model as lm
from sklearn.metrics import accuracy_score
guidedmodel1 = lm.LogisticRegression()
guidedmodel1.fit(Phi_train, Y_train)
y_fitted = guidedmodel1.predict(Phi_train)

training_accuracy = accuracy_score(Y_train.astype(float), y_fitted)


# In[168]:


assert training_accuracy > 0.72


# # Question 6
# 
# That doesn't seem too shabby! But the classifier you made above isn't as good as this might lead us to believe. First, we are evaluating on the training set, which may lead to a misleading accuracy measure, especially if we used the training set to identify discriminative features. In future parts of this analysis, it will be safer to hold out some of our data for model validation and comparison.
# 
# Presumably, our classifier will be used for **filtering**, i.e. preventing messages labelled `spam` from reaching someone's inbox. Since we are trying  There are two kinds of errors we can make:
# - False positive (FP): a ham email gets flagged as spam and filtered out of the inbox.
# - False negative (FN): a spam email gets mislabelled as ham and ends up in the inbox.
# 
# These definitions depend both on the true labels and the predicted labels. False positives and false negatives may be of differing importance, leading us to consider more ways of evaluating a classifier, in addition to overall accuracy:
# 
# **Precision** measures the proportion $\frac{\text{TP}}{\text{TP} + \text{FP}}$ of emails flagged as spam that are actually spam.
# 
# **Recall** measures the proportion $\frac{\text{TP}}{\text{TP} + \text{FN}}$ of spam emails that were correctly flagged as spam. 
# 
# **False-alarm rate** measures the proportion $\frac{\text{FP}}{\text{FP} + \text{TN}}$ of ham emails that were incorrectly flagged as spam. 
# 
# The following image might help:
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Precisionrecall.svg/700px-Precisionrecall.svg.png" width="500px">
# 
# Note that a true positive (TP) is a spam email that is classified as spam, and a true negative (TN) is a ham email that is classified as ham. Answer the following questions in the cells below:
# 
# - (a) Suppose we have a classifier that just predicts 0 (ham) for every email. How many false positives are there? How many false negatives are there? Provide specific numbers using the training data from Question 4.
# - (b) Suppose we have a classifier that just predicts 0 (ham) for every email. What is its accuracy on the training set? What is its recall on the training set?
# - (c) What are the precision, recall, and false-alarm rate of the logistic regression classifier in Question 5? Are there more false positives or false negatives? 
# - (d) Our logistic regression classifier got 75.6% prediction accuracy (number of correct predictions / total). How does this compare with predicting 0 for every email?
# - (e) Given the word features we gave you above, name one reason this classifier is performing poorly.
# - (f) Which of these two classifiers would you prefer for a spam filter and why? (N.B. there is no "right answer" here but be thoughtful in your reasoning).

# In[169]:


# False positive:a ham email gets flagged as spam and filtered out of the inbox.
#False negative: a spam email gets mislabelled as ham and ends up in the inbox.
#provide number of FP and FN, respectively,
# for a classifier that always predicts 0 (never predicts positive...)
zero_predictor_fp = 0
zero_predictor_fn = sum(Y_train)
print(zero_predictor_fp, zero_predictor_fn)


# In[170]:


# This is a cell with just a comment but don't delete me if you want to get credit.


# In[171]:


# provide training accuracy & recall, respectively,
# for a classifier that always predicts 0
z = np.zeros(len(Y_train))
zero_predictor_acc = accuracy_score(z, Y_train)
zero_predictor_recall = 0

print(zero_predictor_acc, zero_predictor_recall)


# In[172]:


# This is a cell with just a comment but don't delete me if you want to get credit.


# In[173]:


# provide training accuracy & recall, respectively,
# for logistic regression classifier from question 5
true_pos = np.sum((Y_train == y_fitted) & y_fitted)
true_neg = np.sum((Y_train == y_fitted) & ~y_fitted)
false_neg = np.sum((Y_train != y_fitted) & ~y_fitted)
false_pos = np.sum((Y_train != y_fitted) & y_fitted)



logistic_predictor_precision = true_pos/(true_pos + false_pos)
logistic_predictor_recall = true_pos/(true_pos + false_neg)
logistic_predictor_far = false_pos/(false_pos + true_neg)

print(logistic_predictor_precision, logistic_predictor_recall, logistic_predictor_far)
print(true_pos, true_neg, false_neg, false_pos)


# In[174]:


# This is a cell with just a comment but don't delete me if you want to get credit.


#     In the all Zero assignment there are no false positives but many false negatives, as everything is assumed to be Ham. The accuracy of using this assignment is actually very close to our method from 5, at ~74%. The word features above cause the classifier to not do well, as it causes many false negatives to occur and has low recall, meaning it is not very good at identifying spam. However, the precision is relatively high still, and very few ham emails were accidentally mislabelled. I would rather have our classifier as opposed to the all zeros one, mostly because I would rather have far less spam and lose a few emails than deal with overwhelming spam in my inbox.

# # Part II - Moving Forward
# 
# With this in mind, it is now your task to make the spam filter more accurate. In order to get full credit on the accuracy part of this assignment, you must get at least **88%** accuracy on the evaluation set. To see your accuracy on the evaluation set, you will use your classifier to predict every email in the `evaluation` DataFrame and upload your predictions to Kaggle.
# 
# To prevent you from fitting to the evaluation set, you may only upload predictions to Kaggle twice per day. This means you should start early and rely on your **test data** to estimate your Kaggle scores.  
# 
# Here are some ideas for improving your model:
# 
# 1. Finding better features based on the email text. Some example features are:
#     1. Number of characters in the subject / body
#     1. Number of words in the subject / body
#     1. Use of punctuation (e.g., how many '!' were there?)
#     1. Number / percentage of capital letters 
#     1. Whether the email is a reply to an earlier email or a forwarded email
# 1. Finding better words to use as features. Which words are the best at distinguishing emails? This requires digging into the email text itself. 
# 1. Better data processing. For example, many emails contain HTML as well as text. You can consider extracting out the text from the HTML to help you find better words. Or, you can match HTML tags themselves, or even some combination of the two.
# 1. Model selection. You can adjust parameters of your model (e.g. the regularization parameter) to achieve higher accuracy. Recall that you should use cross-validation to do feature and model selection properly! Otherwise, you will likely overfit to your training data.
# 
# You may use whatever method you prefer in order to create features. However, **you are only allowed to train logistic regression models and their regularized forms**. This means no random forest, k-nearest-neighbors, neural nets, etc.
# 
# We will not give you a code skeleton to do this, so feel free to create as many cells as you need in order to tackle this task. However, answering questions 7, 8, and 9 should help guide you.
# 
# ---
# 
# **Note:** *You should use the **test data** to evaluate your model and get a better sense of how it will perform on the Kaggle evaluation.*
# 
# ---

# In[179]:


def spamwords(data, words):
    checker = words_in_texts(words, data['email'])
    final=[]
    for text in data['email']:
        feat = [1 if words[i] in text.lower() else 0 for i in range(len(words)) ]
        feat += [len(text.split())]
        feat += [1 if text.count('!') > 5 else 0]
        final.append(pd.Series(feat))
    return np.array(final)


# In[180]:



print(train.columns)



# In[181]:


def rmse(actual, predicted):
    """
    Calculates RMSE from actual and predicted values
    Input:
      actual (1D array-like): vector of actual values
      predicted (1D array-like): vector of predicted/fitted values
    Output:
      a float, the root-mean square error
    """
    rmse = np.sqrt(np.mean(np.square(actual - predicted)))
    return rmse

def select_columns(data, *columns):
    return data.loc[:, columns]

def standardize_columns(data):
    '''
    Input:
      data (data frame): contains only numeric columns
    Output:
      data frame, the same data, except each column is standardized 
      to have 0-mean and unit variance
    '''
    standardize = lambda x: (x-x.mean()) / x.std()
    standardized_data = standardize(data)
    return standardized_data


    
words = ['grants', 'drug', 'bank', 'prescription', 'memo', 'private', 'list', '</div>', 'click', 'unsusbcribe', 'remove', 'free', 'apologies',
              '<html', '=20', '</p>', '=', 'please', 'get', 'one', 'business', 'money', 'here', 'below', '!']



X = spamwords(train, words)
y = train['spam']
X_train, X_test, y_train, y_test = train_test_split(X, y)

final_model = lm.LogisticRegression()
final_model.fit(X_train, y_train)
final_model.score(X_test, y_test)


# # Question 7 (Feature/Model Selection Process)
# 
# In this following cell, describe the process of improving your model. You should use at least 2-3 sentences each to address the follow questions:
# 
# 1. How did you find better features for your model?
# 2. What did you try that worked / didn't work?
# 3. What was surprising in your search for good features?

# 1.) I began by trying different features, such as ratio of capitals, number of words, number of characters, and number of '!'s and honestly found none of these worked terribly well. I then continued to add words to the words list and found that by adding a few more common words found in spam, my model could be very accurate only using the words feature with added specifications.  
# 
# 
# 2.) I tried many words, shoutout to the person who posted a WordCloud of popular spam words on Piazza, which worked and didnt work. But more interestingly, I found that ratios of character counts to number of distinct words differs in spam emails (shown in EDA below), despite being unable to build a reasonable feature to accomodate this. The proportions of capitals also did not increase accuracy, and checking through IDs proved fruitless as well.
# 
# 
# 3.) There are so many complex ways to approach any Data problem like this one, but sometimes the most accurate results come from keeping things simple. I learned that a little EDA can go a long way, and also that gmail's spam filter needs to be improved.

# # Question 8 (EDA)
# 
# In the two cells below, show a visualization that you used to select features for your model. Include both
# 
# 1. A plot showing something meaningful about the data that helped you during feature / model selection.
# 2. 2-3 sentences describing what you plotted and what its implications are for your features.
# 
# Feel to create as many plots as you want in your process of feature selection, but select one for the cells below.
# 
# **You should not show us a visualization just like in question 3.** Specifically, don't show us a bar chart of proportions, or a one-dimensional class conditional density plot. Any other plot is acceptable, as long as it comes with thoughtful commentary. Here are some ideas:
# 
# 1. Consider the correlation between multiple features (look up correlation plots and `sns.heatmap`). 
# 1. Try to show redundancy in a group of features (e.g. `body` and `html` might co-occur relatively frequently, or you might be able to design a feature that captures all html tags and compare it to these). 
# 1. Use a word-cloud or another visualization tool to characterize the most common spam words.
# 1. Visually depict whether spam emails tend to be wordier (in some sense) than ham emails.

# In[ ]:



#sns.kdeplot(train[train['spam']==0]['NumChars'], train[train['spam']==0]['NumWords'], label='Ham')
#sns.kdeplot(train[train['spam']==1]['NumChars'], train[train['spam']==1]['NumWords'], label='Ham')
plt.xlabel = 'Characters in Email'
plt.ylabel = 'Words in Email'
plt.scatter(train[train['spam']==0]['NumChars'], train[train['spam']==0]['NumWords'], label='Ham', color='blue', marker='o', s=20, alpha = 0.1)

plt.legend()
plt.ylim([0,1000])
plt.xlim([0,10000])
plt.show()
plt.scatter(train[train['spam']==1]['NumChars'], train[train['spam']==1]['NumWords'], label='Spam', color = 'red', marker='^', s=20, alpha = 0.1)
plt.legend()
plt.ylim([0,1000])
plt.xlim([0,10000])
plt.show()

#Plots show characters in email texts vs. separate words in email texts




# These plots show characters in email texts vs. separate words in email texts for each Ham and Spam emails respectively. Ham emails seem to follow a largely linear function, while spam emails deviate heavily, especially when characters surpasses 2000. Most spam emails are also concentrated in the 0-1000 character range while ham emails are typically between 0-2500 characters. Spam emails typically have more characters than separate words by a higher margin than ham emails, as exhibited by the higher slope of the ham email values in the plot. However, when trying to implement a feature which took this into account, I failed to make any meaningful progress.

# # Question 9 (Making a Precision-Recall Curve)
# 
# We can trade off between precision and recall. In most cases we won't be able to get both perfect precision (i.e. no false positives) and recall (i.e. no false negatives), so we have to compromise. For example, in the case of cancer screenings, false negatives are comparatively worse than false positives — a false negative means that a patient might not discover a disease until it's too late to treat, while a false positive means that a patient will probably have to take another screening.
# 
# Recall that logistic regression calculates the probability that an example belongs to a certain class. Then, to classify an example we say that an email is spam if our classifier gives it $\ge 0.5$ probability of being spam. However, *we can adjust that cutoff*: we can say that an email is spam only if our classifier gives it $\ge 0.7$ probability of being spam, for example. This is how we can trade off false positives and false negatives.
# 
# The precision-recall curve shows this trade off for each possible cutoff probability. In the cell below, [plot a precision-recall curve](http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#plot-the-precision-recall-curve) for your final classifier (the one you use to make predictions for Kaggle).

# In[ ]:


from sklearn.metrics import precision_recall_curve

# Note that you'll want to use the .predict_proba(...) method for your classifier
# instead of .predict(...) so you get probabilities, not classes
precision, recall, _ = precision_recall_curve(y_train, [c[1] for c in final_model.predict_proba(X_train)])
plt.plot(precision, recall)
plt.ylabel('Recall')
plt.xlabel('Precision')


# # Question 10: Submitting to Kaggle
# 
# The following code will write your predictions on the evaluation dataset to a CSV, which you can submit to Kaggle. You may need to modify it to suit your needs.
# 
# Save your predictions in a 1-dimensional array called `evaluation_predictions`. *Even if you are not submitting to Kaggle, please make sure you've saved your predictions to `evaluation_predictions` as this is how your grade for this part will be determined.*
# 
# Remember that if you've performed transformations or featurization on the training data, you must also perform the same transformations on the evaluation data in order to make predictions. For example, if you've created features for the words "drug" and "money" on the training data, you must also extract the same features in order to use scikit-learn's `.predict(...)` method.
# 
# You should submit your CSV files to https://www.kaggle.com/t/39fae66747b14fd48fe0984f2e4f16ac

# In[183]:


# CHANGE ME (Currently making random predictions)
evaluation_predictions = np.array(final_model.predict(spamwords(evaluation, words)))



# In[184]:


# must be ndarray of predictions
assert isinstance(evaluation_predictions, np.ndarray) 

# must be binary labels (0 or 1) and not probabilities
assert np.all((evaluation_predictions == 0) | (evaluation_predictions == 1))

# must be the right number of predictions
assert evaluation_predictions.shape == (1000, )


# In[127]:


# Please do not modify this cell


# The following saves a file to submit to Kaggle.

# In[ ]:


from datetime import datetime

# Assuming that your predictions on the evaluation set are stored in a 1-dimensional array called
# evaluation_predictions. Feel free to modify this cell as long you create a CSV in the right format.

# must be ndarray of predictions
assert isinstance(evaluation_predictions, np.ndarray) 

# must be binary labels (0 or 1) and not probabilities
assert np.all((evaluation_predictions == 0) | (evaluation_predictions == 1))

# must be the right number of predictions
assert evaluation_predictions.shape == (1000, )

# Construct and save the submission:
submission_df = pd.DataFrame({
    "Id": evaluation['id'], 
    "Class": evaluation_predictions,
}, columns=['Id', 'Class'])
timestamp = datetime.isoformat(datetime.now()).split(".")[0]
submission_df.to_csv("submission_{}.csv".format(timestamp), index=False)

print('Created a CSV file: {}.'.format("submission_{}.csv".format(timestamp)))
print('You may now upload this CSV file to Kaggle for scoring.')

