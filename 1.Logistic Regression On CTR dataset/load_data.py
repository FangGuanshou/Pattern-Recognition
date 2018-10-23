import pandas as pd

#style.use('ggplot')
def load_dataset():
    train_dataset = pd.read_csv('train.csv',parse_dates=True,usecols=range(2,15))
    train_dataset = train_dataset.fillna(0)
    train_set_x_orig = train_dataset.values

    train_dataset = pd.read_csv('train.csv',parse_dates=True,usecols=range(1,2))
    train_dataset = train_dataset.fillna(0)
    train_set_y_orig = train_dataset.values

    test_dataset = pd.read_csv('test.csv',parse_dates=True,usecols=range(1,14))
    test_dataset = test_dataset.fillna(0)
    test_set_x_orig = test_dataset.values
    
    test_dataset = pd.read_csv('submission.csv',parse_dates=True,usecols=range(1,2))
    test_dataset = test_dataset.fillna(0)
    test_set_y_orig = test_dataset.values

    return train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig


if __name__=="__main__":
    train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig = load_dataset()
    print(train_set_x_orig)
    print(train_set_y_orig)
    print(test_set_x_orig)
    print(test_set_y_orig)
    
    print(train_set_y_orig.T.shape[1])
    print(train_set_x_orig.T.shape[1])
    print(test_set_y_orig.T.shape[1])
    print(test_set_x_orig.T.shape[1])
