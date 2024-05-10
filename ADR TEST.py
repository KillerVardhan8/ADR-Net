#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

data = {
    'reaction': ['headache', 'nausea', 'rashes', 'dizziness', 'vomiting',
                 'fatigue', 'insomnia', 'diarrhea', 'constipation', 'blurred vision',
                 'dry mouth', 'itching', 'swelling', 'palpitations', 'irregular heartbeat',
                 'muscle pain', 'joint pain', 'back pain', 'difficulty breathing', 'excessive sweating'],
    'label': ['Not ADR', 'ADR', 'ADR', 'Not ADR', 'ADR',
              'Not ADR', 'Not ADR', 'ADR', 'Not ADR', 'ADR',
              'Not ADR', 'Not ADR', 'ADR', 'ADR', 'ADR',
              'Not ADR', 'Not ADR', 'Not ADR', 'ADR', 'ADR']}

df = pd.DataFrame(data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['reaction'])
y = df['label']

clf = RandomForestClassifier()
clf.fit(X, y)


def predict_adr_probability(condition):
    condition_vec = vectorizer.transform([condition])
    prob = clf.predict_proba(condition_vec)[0]
    return {'ADR': prob[0], 'Not ADR': prob[1]}

user_condition = input("Please enter your condition: ")
prediction_prob = predict_adr_probability(user_condition)
print("Prediction probabilities for user condition '{}':".format(user_condition))
print(prediction_prob)


# In[ ]:




