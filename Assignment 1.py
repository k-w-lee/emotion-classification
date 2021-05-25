#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
os.getcwd()
# Set path for new working directory
path = "C:/Users/Lee Kah Win/Desktop/Master DS Subjects/CDS503 - Machine Learning/Assignment 1/Assignment 1 Data"
os.chdir(path)


# In[2]:


df_freq = pd.read_csv("Freq-PHO-Binary.csv")
df_norm = pd.read_csv("Norm-PHO-Binary.csv")
df_freq.info()
df_norm.info()


# In[3]:


df_freq = df_freq.drop(['Unnamed: 10'], axis=1)
df_freq.head()


# In[4]:


df_norm.head()


# In[5]:


print(df_norm['Depression'].value_counts())
print(df_freq['Depression'].value_counts())


# In[6]:


plt.style.use("fivethirtyeight")
df_freq.Depression.value_counts().plot(kind='bar', title='Freq-PHO-Binary Depression Distribution')


# In[7]:


df_norm.Depression.value_counts().plot(kind='bar', title='Norm-PHO-Binary Depression Distribution')


# # To report distribution for df_norm

# In[8]:


plt.style.use("fivethirtyeight")
x = df_norm['Emotion_Joy']
plt.hist(x, bins=20, edgecolor='black')
plt.xlabel("Emotion_Joy")
plt.ylabel('Frequency')
plt.title(f" Emotion_Joy Histogram Frequency Chart")
plt.show


# In[9]:


x = df_norm['Emotion_Sadness']
plt.hist(x, bins=20, edgecolor='black')
plt.xlabel("Emotion_Sadness")
plt.ylabel('Frequency')
plt.title(f" Emotion_SadnessHistogram Frequency Chart")
plt.show


# In[10]:


x = df_norm['Emotion_Anger']
plt.hist(x, bins=20, edgecolor='black')
plt.xlabel("Emotion_Anger")
plt.ylabel('Frequency')
plt.title(f" Emotion_Anger Histogram Frequency Chart")
plt.show


# In[11]:


x = df_norm['Emotion_Disgust']
plt.hist(x, bins=20, edgecolor='black')
plt.xlabel("Emotion_Disgust")
plt.ylabel('Frequency')
plt.title(f" Emotion_Disgust Histogram Frequency Chart")
plt.show


# In[12]:


x = df_norm['Emotion_Fear']
plt.hist(x, bins=20, edgecolor='black')
plt.xlabel("Emotion_Fear")
plt.ylabel('Frequency')
plt.title(f" Emotion_Fear Histogram Frequency Chart")
plt.show


# In[13]:


x = df_norm['Emotion_Surprise']
plt.hist(x, bins=20, edgecolor='black')
plt.xlabel("Emotion_Surprise")
plt.ylabel('Frequency')
plt.title(f" Emotion_Surprise Histogram Frequency Chart")
plt.show


# In[14]:


x = df_norm['Emotion_Contempt']
plt.hist(x, bins=20, edgecolor='black')
plt.xlabel("Emotion_Contempt")
plt.ylabel('Frequency')
plt.title(f" Emotion_Contempt Histogram Frequency Chart")
plt.show


# In[15]:


x = df_norm['Emotion_Neutral']
plt.hist(x, bins=20, edgecolor='black')
plt.xlabel("Emotion_Neutral")
plt.ylabel('Frequency')
plt.title(f" Emotion_Neutral Histogram Frequency Chart")
plt.show


# # Preprocessing

# ## Preprocessing - Label Encoding

# In[16]:


le = LabelEncoder()
df_norm.Gender = le.fit_transform(df_norm.Gender)
df_norm.Depression = le.fit_transform(df_norm.Depression)
df_freq.Gender = le.fit_transform(df_freq.Gender)
df_freq.Depression = le.fit_transform(df_freq.Depression)
df_norm.head()


# ## Preprocessing -  Balance Data by Random Over Sampling

# In[17]:


count_freq_class_0, count_freq_class_1 = df_freq.Depression.value_counts()
count_norm_class_0, count_norm_class_1 = df_norm.Depression.value_counts()
df_freq_class_0 = df_freq[df_freq['Depression'] == 0]
df_freq_class_1 = df_freq[df_freq['Depression'] == 1]
df_norm_class_0 = df_norm[df_norm['Depression'] == 0]
df_norm_class_1 = df_norm[df_norm['Depression'] == 1]                                                            


# In[18]:


print(df_freq_class_0.shape)
print(df_norm_class_0.shape)
print(df_freq_class_1.shape)
print(df_norm_class_1.shape)


# In[19]:


df_freq_class_1_over = df_freq_class_1.sample(count_freq_class_0, replace=True)
df_norm_class_1_over = df_norm_class_1.sample(count_norm_class_0, replace=True)
print(df_freq_class_1_over.shape)
print(df_norm_class_1_over.shape)


# In[20]:


df_freq = pd.concat([df_freq_class_0,df_freq_class_1_over],axis=0)
df_norm = pd.concat([df_norm_class_0,df_norm_class_1_over],axis=0)
print(df_freq.shape)
print(df_norm.shape)
print(df_freq.Depression.value_counts())
print(df_norm.Depression.value_counts())


# In[21]:


df_freq.Depression.value_counts().plot(kind='bar', title='Freq-PHO-Binary Depression Distribution')


# In[22]:


df_norm.Depression.value_counts().plot(kind='bar', title='Norm-PHO-Binary Depression Distribution')


# ## Preprocessing - One Hot Encoding

# In[23]:


df_norm.head()


# In[24]:


df_norm_enc = pd.get_dummies(df_norm, prefix = ['Gender'], columns = ['Gender'])
df_norm_enc.head()


# In[25]:


df_freq_enc = pd.get_dummies(df_freq, prefix = ['Gender'], columns = ['Gender'])
df_freq_enc.head()


# ## Preprocessing - Separate Target and Features

# In[26]:


features_freq = df_freq_enc.drop('Depression', axis=1)
target_freq = df_freq_enc.Depression
features_norm = df_norm_enc.drop('Depression', axis=1)
target_norm = df_norm_enc.Depression


# ## Preprocessing - Normalisation

# In[27]:


from sklearn import preprocessing

df = preprocessing.normalize(features_freq)
names = features_freq.columns
features_freq = pd.DataFrame(df, columns=names)
features_freq.head()


# In[28]:


d = preprocessing.normalize(features_norm)
names = features_norm.columns
features_norm = pd.DataFrame(d, columns=names)
features_norm.head()


# # Modelling - Baseline

# In[29]:


scoring = ['accuracy','precision_weighted', 'recall_weighted', 'f1_weighted']


# In[30]:


dummy_clf = DummyClassifier(strategy="uniform")
for score_dummy in scoring:
    scoredummy = cross_val_score(dummy_clf, features_freq, target_freq, cv=5,scoring = score_dummy)
    print(f"Freq-PHO-Binary - Dummy - {str(score_dummy)} : {scoredummy.mean()}")


# In[31]:


for score_dummy in scoring:
    scoredummy = cross_val_score(dummy_clf, features_norm, target_norm, cv=5,scoring = score_dummy)
    print(f"Norm-PHO-Binary - Dummy - {str(score_dummy)} : {scoredummy.mean()}")


# # Modelling - Support Vector Machine

# In[32]:


models = [SVC(kernel='linear'), SVC(kernel='poly'), SVC(kernel='rbf'), SVC(kernel='sigmoid')]


# In[66]:


for model in models:
    if model == models[0]:
        param_grid = {'C': [0.01,0.0125,0.02,0.03,0.04,0.05 ]}
    elif model == models[1]:
        param_grid = {'C': [0.01,0.1,1, 10, 25, 50,100, 1000,10000], 'degree': [2, 3, 4, 5], 'gamma': ['scale','auto']}
    elif model == models[2]:
        param_grid = {'C': [0.5,0.75,1, 1.25,1.5,1.75,2],'gamma': ['scale','auto'],'kernel': ['rbf']}
    else:
        param_grid = {'C': [0.1,0.125,0.2,0.3,0.4 ], 'gamma': ['scale','auto']}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='recall_weighted', n_jobs=-1)
    grid_result = grid_search.fit(features_norm, target_norm)
    print(str(model), '-', grid_result.best_params_)
    print(grid_result.best_score_)
    print('')


# In[67]:


for model in models:
    if model == models[0]:
        param_grid = {'C': [9000,10000,20000]}
    elif model == models[1]:
        param_grid = {'C': [75,78,80,82,85,90], 'degree': [ 5,6,7,8,9,10], 'gamma': ['scale','auto']}
    elif model == models[2]:
        param_grid = {'C': [950,975, 1000,1025,1050], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                      'gamma': ['scale','auto'],'kernel': ['rbf']}
    else:
        param_grid = {'C': [1200,1300,2000,3000], 'gamma': ['scale','auto']}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='recall_weighted', n_jobs=-1)
    grid_result = grid_search.fit(features_freq, target_freq)
    print(str(model), '-', grid_result.best_params_)
    print(grid_result.best_score_)
    print('')


# In[57]:


svc_freq=SVC(kernel='poly', C=85, degree = 5, gamma= 'scale')
accuracy = cross_val_score(svc_freq, features_freq, target_freq, cv=5,scoring = 'accuracy')
precision_weighted = cross_val_score(svc_freq, features_freq, target_freq, cv=5,scoring = 'precision_weighted')
recall_weighted = cross_val_score(svc_freq, features_freq, target_freq, cv=5,scoring = 'recall_weighted')
f1_weighted = cross_val_score(svc_freq, features_freq, target_freq,cv=5, scoring = 'f1_weighted')
print(f"Accuracy : {accuracy.mean()}")
print(f"Precision_weighted : {precision_weighted.mean()}")
print(f"Recall_weighted :{recall_weighted.mean()}")
print(f"f1_weighted :{f1_weighted.mean()}")


# In[58]:


svc_norm=SVC(kernel='poly', C= 50, degree= 5, gamma= 'scale')
accuracy = cross_val_score(svc_norm, features_norm, target_norm, cv=5,scoring = 'accuracy')
precision_weighted = cross_val_score(svc_norm, features_norm, target_norm, cv=5,scoring = 'precision_weighted')
recall_weighted = cross_val_score(svc_norm, features_norm, target_norm, cv=5,scoring = 'recall_weighted')
f1_weighted = cross_val_score(svc_norm, features_norm, target_norm,cv=5, scoring = 'f1_weighted')
print(f"Accuracy : {accuracy.mean()}")
print(f"Precision_weighted : {precision_weighted.mean()}")
print(f"Recall_weighted :{recall_weighted.mean()}")
print(f"f1_weighted :{f1_weighted.mean()}")


# # Modelling - KNN

# In[37]:


# instantiate model
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, features_freq, target_freq, cv=5, scoring='accuracy')
print('Accuracy (Validation) =', scores.mean())     


# In[38]:


# define the parameter values that should be searched
# for python 2, k_range = range(1, 31)
k_range = list(range(1, 219))
weight_options = ["uniform", "distance"]


# In[68]:


# The param_grid specifies one grid should be explored
param_grid = dict(n_neighbors = k_range, weights = weight_options)                                            
# Fit on the dataset on all parameter combinations in param_grid
# Retain the best combination
grid_search_freq = GridSearchCV(knn, param_grid, cv = 5, scoring = 'recall_weighted')
grid_search_freq.fit(features_freq, target_freq)
print(grid_search_freq.best_params_)
print(grid_search_freq.best_score_)


# In[69]:


knn_freq = KNeighborsClassifier(n_neighbors=192, weights = 'distance')
for score_knn in scoring:
    scoreknn = cross_val_score(knn_freq, features_freq, target_freq, cv=5,scoring = score_knn)
    print(f"Freq-PHO-Binary - KNN - {str(score_knn)} : {scoreknn.mean()}")


# In[70]:


grid_search_norm = GridSearchCV(knn, param_grid, cv = 5, scoring = 'recall_weighted')
grid_search_norm.fit(features_norm, target_norm)
print(grid_search_norm.best_params_)
print(grid_search_norm.best_score_)


# In[71]:


knn_norm = KNeighborsClassifier(n_neighbors=199, weights = 'distance')
for score_knn in scoring:
    scoreknn = cross_val_score(knn_norm, features_norm, target_norm, cv=5,scoring = score_knn)
    print(f"Norm-PHO-Binary - KNN - {str(score_knn)} : {scoreknn.mean()}")


# # Modelling - Decision Tree Classifier

# In[43]:


dtc = DecisionTreeClassifier(min_samples_split=2, random_state=0)
# Define the parameter values that should be searched
sample_split_range = list(range(2, 10))
criterion_options = ['gini', 'entropy']
max_depth_options = list(range(2, 50))
num_leafs = [1, 5, 10, 20, 50]


# In[44]:


param_grid = dict(min_samples_split=sample_split_range, 
                  criterion = criterion_options, max_depth = max_depth_options,
                 min_samples_leaf = num_leafs)
for score in scoring:
    grid_freq = GridSearchCV(dtc, param_grid, cv=5, scoring=score)
    grid_freq.fit(features_freq, target_freq)
    print("Freq-PHO-Binary")
    print(grid_freq.best_params_)
    print(str(score))
    print(grid_freq.best_score_)


# In[54]:


# due to our selected criteria is recall, therefore the recall optimised parameter is used
dtc_freq = DecisionTreeClassifier(min_samples_split=2, criterion= 'gini', 
                                  max_depth= 10, min_samples_leaf= 1,random_state=0)
for score_dtc in scoring:
    scoredtc = cross_val_score(dtc_freq, features_freq, target_freq, cv=5,scoring = score_dtc)
    print(f"Freq-PHO-Binary - DTC - {str(score_dtc)} : {scoredtc.mean()}")


# In[45]:


for score in scoring:
    grid_norm = GridSearchCV(dtc, param_grid, cv=5, scoring=score)
    grid_norm.fit(features_norm, target_norm)
    print("Norm-PHO-Binary")    
    print(grid_norm.best_params_)
    print(str(score))
    print(grid_norm.best_score_)


# In[56]:


dtc_norm = DecisionTreeClassifier(min_samples_split=2, criterion= 'gini', 
                                  max_depth= 12, min_samples_leaf= 1,random_state=0)
for score_dtc in scoring:
    scoredtc = cross_val_score(dtc_norm, features_norm, target_norm, cv=5,scoring = score_dtc)
    print(f"Norm-PHO-Binary - DTC - {str(score_dtc)} : {scoredtc.mean()}")


# # Modelling - RandomForestClassifier

# In[76]:


rfc=RandomForestClassifier(random_state=0)
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

CV_rfc_freq = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5, scoring = 'recall_weighted')
CV_rfc_freq.fit(features_freq, target_freq)
CV_rfc_freq.best_params_


# In[77]:


CV_rfc_freq=RandomForestClassifier(random_state=0, max_features='auto', n_estimators= 200, 
                                   max_depth=8, criterion='gini')
for score_rfc in scoring:
    scorerfc = cross_val_score(CV_rfc_freq, features_freq, target_freq, cv=5,scoring = score_rfc)
    print(f"Freq-PHO-Binary - RFC - {str(score_rfc)} : {scorerfc.mean()}")


# In[78]:


CV_rfc_norm = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5, scoring = 'recall_weighted')
CV_rfc_norm.fit(features_norm, target_norm)
CV_rfc_norm.best_params_


# In[80]:


CV_rfc_norm=RandomForestClassifier(random_state=0, max_features='auto', n_estimators= 200, 
                                   max_depth=7, criterion='entropy')
for score_rfc in scoring:
    scorerfc = cross_val_score(CV_rfc_norm, features_norm, target_norm, cv=5,scoring = score_rfc)
    print(f"Norm-PHO-Binary - RFC - {str(score_rfc)} : {scorerfc.mean()}")


# # Save the best performing model

# In[75]:


# Import pickle
import pickle
# Specify the file name to save the model
# Use filename='freq_model.sav' for Freq-PHO-Binary
# Use filename='norm_model.sav' for Norm-PHO-Binary
filename='norm_model.sav'
# Open the file name in write mode. Pass the filename and model.
# Replace modelname with the name of your model
pickle.dump(knn_norm, open(filename, 'wb'))

