#%%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier #RandomForestClassifier(max_depth = 2, random_state=0)
from sklearn.linear_model import LinearRegression #LinearRegression()
from sklearn.neural_network import MLPClassifier #MLPClassifier()
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis #QuadraticDiscriminantAnalysis()
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

##read needed data from provided Excel file and save it to .csv files
#data_xls = pd.read_excel('dane.xlsx', 'ucz', index_col=0)
#data_xls.to_csv('dane_ucz.csv', encoding='utf-8', index=True)
#data_xls = pd.read_excel('dane.xlsx', 'test', index_col=0)
#data_xls.to_csv('dane_test.csv', encoding='utf-8', index=True)

#names from models variable are used for output filenames and console debug
models = ['random_forest', 'linear_regression', 'mlp', 'quadratic_discriminant'] # 0, 1, 2, 3

#model is an index number of model from models, modify it to choose a desired model
model = 3

#select desired model
if model == 0:
    model_name = RandomForestClassifier(max_depth = 2, random_state=0)
elif model == 1:
    model_name = LinearRegression()
elif model == 2:
    model_name = MLPClassifier()
elif model == 3:
    model_name = QuadraticDiscriminantAnalysis()
    
#specify output filename based on chosen model
file_name = models[model]

#list of data I need to use
columns = ['Y', 'X1_Z2', 'X4_Z1', 'Ranga(X5)', 'X6_Z2', 'X6_Z3', 'X6_Z5', 'X9', 'X10_Z2', 'X10_Z3',
        'Ranga(X11)', 'X12_Z1', 'Ranga(X12)', 'X14_Z2', 'Ranga(X14)',  'X15_Z1',
        'X15_Z3', 'X18', 'X24']

#read all data from .csv file we made earlier (note - the train data here was previously stripped of outliers)
train_dataset = pd.read_csv('dane_ucz.csv')
test_dataset = pd.read_csv('dane_test.csv')

#extract the data I need from the columns list
train_data = train_dataset[columns]
test_data = test_dataset[columns]

#extract target from data
train_y = np.array(train_data['Y'])
test_y = np.array(test_data['Y'])

#extract features from data
train_X = train_data.drop('Y', axis=1)
test_X = test_data.drop('Y', axis=1)

#convert features to numpy array
train_X = np.array(train_X)
test_X = np.array(test_X)

#train model
trained_model = model_name.fit(train_X, train_y)

#read the raw train data (with outliers)
train_predict_dataset = pd.read_csv('dane_ucz_2.csv')
train_predict_data = train_predict_dataset[columns]
train_predict_y = np.array(train_predict_data['Y'])
train_predict_X = train_predict_data.drop('Y', axis=1)
train_predict_X = np.array(train_predict_X)

#make predictions on test data and raw train data
predictions_test = trained_model.predict(test_X)
predictions_train = trained_model.predict(train_predict_X)

#round the predictions to 0 or 1, because the target is either 0 or 1
predictions_test[predictions_test < 0.5] = 0
predictions_test[predictions_test >= 0.5] = 1
predictions_train[predictions_train < 0.5] = 0
predictions_train[predictions_train >= 0.5] = 1

#create an array with actual target and predicted target
results_test = np.c_[test_y, predictions_test]
results_train = np.c_[train_predict_y, predictions_train]

#save the array to a .csv file
pd.DataFrame(results_test).to_csv(f'{file_name}_test.csv', encoding='utf-8')
pd.DataFrame(results_train).to_csv(f'{file_name}_train.csv', encoding='utf-8')

#calculate the predictions accuracy
train_accuracy = accuracy_score(train_predict_y, predictions_train)
test_accuracy = accuracy_score(test_y, predictions_test)

#create confusion matricies
train_matrix = confusion_matrix(train_predict_y, predictions_train)
test_matrix = confusion_matrix(test_y, predictions_test)

#write model performance to console
print(models[model])
print(f'Train accuracy: {round(train_accuracy*100, 2)}%')
print(f'Test accuracy: {round(test_accuracy*100, 2)}%')
print('Confusion matrix for train data')
print(train_matrix)
print('Confusion matrix for test data')
print(test_matrix)
# %%
