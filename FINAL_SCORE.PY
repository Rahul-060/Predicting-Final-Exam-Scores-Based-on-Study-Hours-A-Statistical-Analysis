# Predict a student's final exam score based on the number of hours they study

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# step-2 data filling 
data={'Hour_study':[2,3,4,5,6,7,8,9,10],'Exam_score':[50,60,70,75,80,85,90,95,99]}

# step-3 representation of our data into dataframe
df=pd.DataFrame(data)
# print(df)

# step-4
X=df[['Hour_study']]
y=df[['Exam_score']]

# step-5 train and test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42) # 20% will be tested and 80% will be trained
# random_state will be set at 42 so that our data reproductivity will be remained and we get the same train and test sets across different executions

# step-6 create an instance
model=LinearRegression()

# step-7 
model.fit(X_train,y_train)

# user input testing
user_input=float(input(("Enter the number of hours you study: ")))

# Now predicting by data provided by user
predicted_score=model.predict([[user_input]])

# printing the output
print("Predicted exam score: ")
print(predicted_score[0:2])
