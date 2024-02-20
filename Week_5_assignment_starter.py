#!/usr/bin/env python
# coding: utf-8

# # DS Automation Assignment

# Using our prepared churn data from week 2:
# - use pycaret to find an ML algorithm that performs best on the data
#     - Choose a metric you think is best to use for finding the best model; by default, it is accuracy but it could be AUC, precision, recall, etc. The week 3 FTE has some information on these different metrics.
# - save the model to disk
# - create a Python script/file/module with a function that takes a pandas dataframe as an input and returns the probability of churn for each row in the dataframe
#     - your Python file/function should print out the predictions for new data (new_churn_data.csv)
#     - the true values for the new data are [1, 0, 0, 1, 0] if you're interested
# - test your Python module and function with the new data, new_churn_data.csv
# - write a short summary of the process and results at the end of this notebook
# - upload this Jupyter Notebook and Python file to a Github repository, and turn in a link to the repository in the week 5 assignment dropbox
# 
# *Optional* challenges:
# - return the probability of churn for each new prediction, and the percentile where that prediction is in the distribution of probability predictions from the training dataset (e.g. a high probability of churn like 0.78 might be at the 90th percentile)
# - use other autoML packages, such as TPOT, H2O, MLBox, etc, and compare performance and features with pycaret
# - create a class in your Python module to hold the functions that you created
# - accept user input to specify a file using a tool such as Python's `input()` function, the `click` package for command-line arguments, or a GUI
# - Use the unmodified churn data (new_unmodified_churn_data.csv) in your Python script. This will require adding the same preprocessing steps from week 2 since this data is like the original unmodified dataset from week 1.

# # Load Data 

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('clean_churn_data.csv')


# In[3]:


df.head()


# In[4]:


df.isna().sum()


# In[5]:


df_copy = df.copy()


# In[6]:


# removing avgmonthly charge
# column as a whole as its not part of the orginal data provided and its similar to monthly charges
df.drop(['AvgMonthlyCharges', 'Unnamed: 0','customerID'], axis = 1, inplace = True)


# In[7]:


df


# # use pycaret to find an ML algorithm that performs best on the data

# In[8]:


from pycaret.classification import *


# In[9]:


automl = setup(df, target = 'Churn')


# # Choose a metric you think is best to use for finding the best model; by default, it is accuracy but it could be AUC, precision, recall, etc. The week 3 FTE has some information on these different metrics.

# In[10]:


best_model = compare_models( fold = 2, sort = 'Recall')


# # save the model to disk

# In[11]:


save_model(best_model, 'best_model_recall')


# # create a Python script/file/module with a function that takes a pandas dataframe as an input and returns the probability of churn for each row in the dataframe
# ## your Python file/function should print out the predictions for new data (new_churn_data.csv)
# ## the true values for the new data are [1, 0, 0, 1, 0] if you're interested

# In[12]:


import pickle

def prob_churn(data):
    
    with open ('best_model_recall.pk','rb') as f:
            loaded_model = pickle.load(f)
    
    loaded_lda = load_model('best_model_recall')
    
    
    prediction = predict_model(loaded_lda, data)
    
    return prediction

data = pd.read_csv('new_churn_data.csv')

output = prob_churn(data)
#output_with_id = pd.concat([df_copy['customerID'], output], axis=1)

print(output)
        


# In[13]:


print("True values for new data:")
print(output['prediction_label'] )

print("\n Expected True values for new data:")
true_values = [1, 0, 0, 1, 0]
print("\n ",true_values)


# # Summary

# For this assignment, I started by loading my data and checking for any missing values in case I missed any from the week 2 assignment file. It seemed that I had some missing files, so I decided to drop the column "AvgMonthlyCharges" as it had missing values as well as having similar content as the "MonthlyCharges" column. After completing that step, I went on to drop additional columns like "uname: 0" as it wasn't needed, as well as "CustomerID" as sometimes it causes issues in my code when wanting to only deal with integer values but the customer ID contains both character and integer values.
# 
# After having a clean dataset, I went on to use PyCaret's AutoML to run my machine learning algorithm. Using the setup function, I specified my data and my target variable which was "Churn." Afterwards, I ran a compare model test with the metric set as recall because recall is the best model when looking at churn because it can best prioritize the identification of churn cases, aiming to minimize false negatives and capture as many actual churners as possible.
# 
# Once completing that step, I went on to save my model to disk for later use. In the end, to use it on new data, I retrieved it in my function called "prob_churn" using the open statement as a readable file only. Once it did that, I loaded the model to be used using the "load_model" function. Afterward, in my function, I was able to use the "predict_model" function to use my loaded model to predict the outcome of the new given data.
# 
# You can see from the result of using "new_churn_data.csv" as my data, I get a true value of 10001 instead of the proposed true value of 10001. My guess for the reason being is that "fold = 5" might be affecting the true positive, but I did try changing it and still got an outcome of 10101 instead using "fold = 20."

# Write a short summary of the process and results here.
