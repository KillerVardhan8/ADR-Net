#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

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

lda = LatentDirichletAllocation(n_components=5, random_state=42)
X_lda = lda.fit_transform(X)

lsa = TruncatedSVD(n_components=5, random_state=42)
X_lsa = lsa.fit_transform(X)

X_train_lsa, X_test_lsa, y_train, y_test = train_test_split(X_lsa, y, test_size=0.2, random_state=42)
X_train_lda, X_test_lda, _, _ = train_test_split(X_lda, y, test_size=0.2, random_state=42)

models = {
    'Random Forest (LSA)': RandomForestClassifier(),
    'Random Forest (LDA)': RandomForestClassifier(),
    'Logistic Regression (LSA)': LogisticRegression(),
    'Logistic Regression (LDA)': LogisticRegression()
}

f1_scores = {}
for name, model in models.items():
    if 'LSA' in name:
        X_train, X_test = X_train_lsa, X_test_lsa
    else:
        X_train, X_test = X_train_lda, X_test_lda
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1_scores[name] = f1_score(y_test, y_pred, average='weighted')

plt.bar(f1_scores.keys(), f1_scores.values(), color=['blue', 'orange', 'green', 'red'])
plt.xlabel('Models')
plt.ylabel('F1 Score')
plt.title('F1 Score Comparison for ADR Prediction Models')
plt.ylim(0, 1)  
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[ ]:




