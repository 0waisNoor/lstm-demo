##########################################################################################################################################

##My assumption is that this model is going to be highly inaccurate as it does not realize that the winning must be either team1 or team2##
##even if I change the results to team 1 or 2, the model will seem accurate due to the 50/50 chance

############################################################################################################################################

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import onehotencoding
import openpyxl
#import the dataset
data = pd.read_csv("ODI_match_data.csv")
print(data.columns)

#choose the most important features
features = ['season','gender','team1','team2']
X = data[features]
y = data['winner']

#check for missing values
missing_counts = data[features].isna().sum()
print("missing values:\n", missing_counts)

#extract only the years from the season column
temp = []
for val in X['season']:
    temp.append(val[0:4])
X['season'] = temp

#check for data type anomalies
for feature in features:
    print(data[feature].unique())

#one hot encode categorical fields
season = onehotencoding.oneHotEncodeColumn(X,'season')
gender = onehotencoding.oneHotEncodeColumn(X,'gender')
team1 = onehotencoding.oneHotEncodeColumn(X,'team1')
team2 = onehotencoding.oneHotEncodeColumn(X,'team2')

X = pd.DataFrame({"season":season,'gender':gender,'team1':team1,'team2':team2})

#fill missing values in y variable
y = y.fillna('draw')

#split the data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#create the model
model = DecisionTreeClassifier(random_state=1)

#train the model
print(X['season'])
model.fit(X_train,y_train)

#predict the data
results = model.predict(X_test)

results_df = pd.DataFrame({"season":X_test['season'],"gender":X_test['gender'],"team1":X_test['team1'],"team2":X_test['team2'],"actual":y_test,"predicted":results})
results_df.to_excel("results.xlsx")