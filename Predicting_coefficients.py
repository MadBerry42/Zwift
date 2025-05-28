import numpy as np
import pandas as pd
from sklearn import svm
import sklearn
import sklearn.multioutput
from sklearn.preprocessing import MinMaxScaler

path = r"C:\Users\maddalb\NTNU\Julia Kathrin Baumgart - Protocol Data"

# Import Coefficients Database
database = pd.read_excel(f"{path}\\Coefficients_raw_filtered.xlsx", usecols=["ID", "Age", "Gender", "Height", "Weight", "Alpha mav", "Beta mav"])
database = database.dropna()

participants = [0, 2 , 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 20]

#--------------------------------------------------------------------------------------------------------------------------------------------
    # RapidForest
#--------------------------------------------------------------------------------------------------------------------------------------------
for i, ID in enumerate(participants): 
    # Data preprocessing: division in training and test set, normalization
    X_training = database.loc[database["ID"] != ID, ["Age", "Gender", "Height", "Weight"]]
    y_training = database.loc[database["ID"] != ID, ["Alpha mav", "Beta mav"]]
    X_test = database.loc[database["ID"] == ID, ["Age", "Gender", "Height", "Weight"]]
    y_test = np.array(database.loc[database["ID"] == ID, ["Alpha mav", "Beta mav"]])

    scaler_y = MinMaxScaler()
    scaler_x = MinMaxScaler()
    X_train = pd.DataFrame(scaler_x.fit_transform(X_training), columns = X_training.columns)
    X_test = pd.DataFrame(scaler_x.transform(X_test), columns= X_training.columns)
    y_train = pd.DataFrame(scaler_y.fit_transform(y_training), columns = y_training.columns)
    # y_test = pd.DataFrame(scaler_y.transform(y_test), columns= y_training.columns)

    print('RandomForest tuning')

    model = sklearn.multioutput.MultiOutputRegressor(sklearn.ensemble.RandomForestRegressor(n_estimators = 100, random_state = 42))
    model.fit(X_train, y_train)

    # Predict output
    predicted = model.predict(X_test)
    predicted = pd.DataFrame(scaler_y.inverse_transform(predicted))

    if i == 0:
        df = pd.DataFrame({"Method": "RandomForest",
                           "ID":f"{ID:03}", 
                           "Alpha predicted": predicted[0], "Actual alpha": y_test["Alpha mav"],
                           "Beta predicted": predicted[1], "Actual beta": y_test["Beta mav"]
                           })
    else:
        tmp = pd.DataFrame({"Method": " ",
                           "ID":f"{ID:03}", 
                           "Alpha predicted": predicted[0], "Actual alpha": y_test["Alpha mav"],
                           "Beta predicted": predicted[1], "Actual beta": y_test["Beta mav"]
                           })
        df = pd.concat([df, tmp], axis = 0)


#-----------------------------------------------------------------------------------------------------------------------------------------------
    # SVR
#-----------------------------------------------------------------------------------------------------------------------------------------------
for i, ID in enumerate(participants):
    
    # Data preprocessing: division in training and test set, normalization
    X_training = database.loc[database["ID"] != ID, ["Age", "Gender", "Height", "Weight"]]
    y_training = database.loc[database["ID"] != ID, ["Alpha mav", "Beta mav"]]
    X_test = database.loc[database["ID"] == ID, ["Age", "Gender", "Height", "Weight"]]
    y_test = database.loc[database["ID"] == ID, ["Alpha mav", "Beta mav"]]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train = pd.DataFrame(scaler_x.fit_transform(X_training), columns = X_training.columns)
    X_test = pd.DataFrame(scaler_x.transform(X_test), columns= X_training.columns)
    y_train = pd.DataFrame(scaler_y.fit_transform(y_training), columns = y_training.columns)
    y_test = pd.DataFrame(scaler_y.transform(y_test), columns= y_training.columns)

    print('SVR hyper parameter tuning')

    # Define the model (without specifying kernel yet)
    model = svm.SVR()
    model = sklearn.multioutput.MultiOutputRegressor(model)

    # Define hyperparameters to search, including different kernels
    param_grid = {
    'estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Try different kernels
    'estimator__C': [0.1, 1, 10, 100],  # Regularization parameter
    'estimator__gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Kernel coefficient
    'estimator__epsilon': [0.1, 0.2, 0.5],  # Epsilon in the epsilon-SVR
    }

    # Perform Grid Search with Cross-Validation
    n_splits = 3

    # grid_search = sklearn.model_selection.GridSearchCV(estimator=model, param_grid=param_grid, cv=n_splits, n_jobs=-1)
    grid_search = sklearn.model_selection.RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=n_splits, n_jobs=-1, n_iter=50)

    # Fit the model to the data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    model = best_model
    model.fit(X_train, y_train)

    # Predict results
    predicted = model.predict(X_test)

    # Recale
    prediction = pd.DataFrame(scaler_y.inverse_transform(predicted))

    # Print the best parameters
    # print(f"Best hyperparameters: {best_params}")

    #  View cross-validation fold results
    cv_results = grid_search.cv_results_

    alpha_orig = np.array(database.loc[database["ID"] == ID, ["Alpha mav"]]).flatten()
    beta_orig = np.array(database.loc[database["ID"] == ID, ["Beta mav"]]).flatten() 
    

    if i == 0:
        tmp = pd.DataFrame({"Method": "SVR",
                        "ID":f"{ID:03}", 
                        "kernel": best_params["estimator__kernel"], 
                        "Alpha predicted": prediction[0], "Actual alpha": alpha_orig, 
                        "Beta predicted": prediction[1], "Actual beta": beta_orig})
    else:
        tmp = pd.DataFrame({"Method": " ",   
                            "ID":f"{ID:03}", 
                            "kernel": best_params["estimator__kernel"], 
                            "Alpha predicted": prediction[0], "Actual alpha": alpha_orig, 
                            "Beta predicted": prediction[1], "Actual beta": beta_orig})
        df = pd.concat([df, tmp], axis = 0)

print(df)

final = "yake"