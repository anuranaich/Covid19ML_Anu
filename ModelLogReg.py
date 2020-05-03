import numpy as nm 
import matplotlib.pyplot as mtp 
import pandas as pd
import pickle

# Importing the dataset
dataset = pd.read_csv('dataset_Covid_reg.csv')

dataset['Diabetes'].fillna(0, inplace=True) 
dataset['HeartProblem'].fillna(0, inplace=True) 
dataset['asthma'].fillna(0, inplace=True) 

reqdataset = dataset[['Age', 'Diabetes', 'HeartProblem', 'asthma', 'severity_rate']]

X = reqdataset.iloc[:, :4] 
y = reqdataset.iloc[:, -1]



# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(X, y, test_size= 0.25, random_state=0)  


##feature Scaling  
#from sklearn.preprocessing import StandardScaler    
#st_x= StandardScaler()    
#x_train= st_x.fit_transform(x_train)    
#x_test= st_x.transform(x_test)  



from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
 


#Fitting model with trainig data
regressor.fit(x_train, y_train)



# Predicting the Test set results
y_pred = regressor.predict(x_test)


# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[49, 0, 0, 0]]))

predicted= regressor.predict([[100, 1, 1, 1]]) # 0:Overcast, 2:Mild
print(predicted)

print(regressor.score(x_test, y_test))