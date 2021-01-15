#The following function performs stratified sampling. 
#data: DataFrame
#y: string, name of the label attribute
#train_fraction: float, fraction of the training set, e.g. 0.8
#We assume that y has two values 0,1. 
def stratified_sample(data,y,train_fraction):
    import pandas as pd
    data1=data[data[y]==1] 
    data0=data[data[y]==0] 

    train1=data1.sample(frac=train_fraction,random_state=42) 
    test1=data1.drop(train1.index)

    train0=data0.sample(frac=train_fraction,random_state=42) 
    test0=data0.drop(train0.index)

    train=pd.concat([train1,train0]) 
    test=pd.concat([test1,test0])

    X_train = train.drop([y], axis = 1) 
    y_train = train[y] 

    X_test = test.drop([y], axis = 1) 
    y_test = test[y] 
    
    return X_train, X_test, y_train, y_test


#The following function converts a list of categorical variables to dummies (0/1).
#data: DataFrame, contain the entire data with bot numeric and categorical 
def cat_to_dummy(data,list):
    import pandas as pd
    
    for i in list:
        data=pd.concat([data, pd.get_dummies(data[i], prefix = i)], axis = 1)
    data=data.drop(list, axis = 1)
    
    return data

def logistic_function(X_train,y_train,X_test,y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report

    model = LogisticRegression()
    logistic_regression = GridSearchCV(model,param_grid={"penalty": ['l1','l2']})
    logistic_regression.fit(X_train, y_train)
    y_pred = logistic_regression.predict(X_test)

    accuracy = logistic_regression.score(X_test, y_test)
    classification_report = classification_report(y_test, y_pred)
    coefficient = logistic_regression.best_estimator_.coef_
    intercept = logistic_regression.best_estimator_.intercept_

    print(accuracy)
    print(classification_report)
    print(intercept)
    print(coefficient)
    print(X_train.columns)