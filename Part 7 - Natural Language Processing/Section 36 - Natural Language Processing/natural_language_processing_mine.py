# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
"""
##################################################
#cleaning the text 
import re #text treatment
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0]) #keep what caracter we want
review = review.lower() #remove capital letters
review = review.split() #split string in list of words

import nltk #removing useless world
nltk.download('stopwords')
from nltk.corpus import stopwords #putting the download package available for use
#first test
#review = [word for word in review if not word in set(stopwords.words('english'))]#set is used to make it quixk

#will for example make love ~ loved loving;..
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

#coller les mots séparés
review = ' '.join(review)

##################################################
"""
import re #text treatment
import nltk #removing useless world
nltk.download('stopwords')
from nltk.corpus import stopwords #putting the download package available for use
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #keep what caracter we want
    review = review.lower() #remove capital letters
    review = review.split() #split string in list of words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

#creating the bag of word model
from sklearn.feature_extraction.text import CountVectorizer#creating the sparse matrix 
#sparse matric is the matrix of the relevant word per review
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()#to array to convert in matrix for python
y = dataset.iloc[:, 1].values    
    

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)

#uneccesary preprocessing here (Naiv classifcation)
"""# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""

def classifier(i):
    
    if i == 'N':
        ####Naiv#####
        # Fitting classifier to the Training set
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)    
     
    if i == 'RF' : 
        #####Random FOrest #####
        # Fitting classifier to the Training set
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 5, criterion = 'entropy')
        classifier.fit(X_train, y_train)
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)     
 
    if i == 'TD' :   
        ###Tree Decision####
        # Fitting Decision Tree Classification to the Training set
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
      
    if i == 'SVM' :   
        ###SVM###
        # Fitting classifier to the Training set
        from sklearn.svm import SVC
        classifier = SVC(kernel='linear', random_state = 0)
        classifier.fit(X_train, y_train)
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred) 
        
    TP = cm[0][0]
    TN = cm[1][1]
    FP = cm[0][1]
    FN = cm[1][0]
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * precision *recall / (precision + recall)
    
    return accuracy, precision, recall, f1_score

    
    
    
    
    

