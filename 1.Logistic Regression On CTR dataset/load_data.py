# _*_ coding = utf-8 _*_
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

def load_dataset():
    train_dataset = pd.read_csv('train.csv',parse_dates=True,usecols=range(1,15))
    train_dataset = train_dataset.dropna(thresh=11)
    for column in list(train_dataset.columns):
        mean_val = train_dataset[column].mean()
        train_dataset[column].fillna(mean_val,inplace=True)
    train_dataset = train_dataset.values    

    train_set_x_orig = train_dataset[:,1:15]
    train_set_y_orig = train_dataset[:,0:1]

    test_dataset1 = pd.read_csv('test.csv',parse_dates=True,usecols=range(0,14))
    test_dataset2 = pd.read_csv('submission.csv',parse_dates=True,usecols=range(0,2))
    test_dataset = pd.merge(test_dataset2,test_dataset1,on='Id')

    test_dataset = test_dataset.dropna(thresh=11)
    for column in list(test_dataset.columns):
        mean_val = test_dataset[column].mean()
        test_dataset[column].fillna(mean_val,inplace=True)
    test_dataset = test_dataset.values    
    
    print(test_dataset)

    test_set_x_orig = test_dataset[:,2:16]
    test_set_y_orig = test_dataset[:,1:2]

    
    return train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig

if __name__=="__main__":
    train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig = load_dataset()
    print(train_set_x_orig)
    print(train_set_y_orig)
    print(test_set_x_orig)
    print(test_set_y_orig)
    

    print(train_set_x_orig.shape)
    print(train_set_y_orig.shape)
    print(test_set_x_orig.shape)
    print(test_set_y_orig.shape)
    
