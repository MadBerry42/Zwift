import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Modeling
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn import svm
# Data preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
# Evaluating performances
from sklearn.metrics import r2_score
from matplotlib.lines import Line2D

class RPEModel():
    def __init__(self, n_windows:int, participants):
        self.n_windows = n_windows
        self.len_window = 180/n_windows
        self.participants = participants

    def preprocessing(self, data_or):
        data = data_or.drop(columns = ["ID", "RPE"])
        RPE_or = data_or.iloc[:, [5]]
        return data, RPE_or

    def plot_results(y_test, y_test_fit, y_train, y_train_fit, r2_test, mse, rmse, r2_train, mse_train, rmse_train, CV_suffix = ""):
        # Create a figure with two subplots (2 row, 2 columns)
        fig, axes = plt.subplots(2, 2, figsize=(14, 6))
        axes = axes.flatten()  # Flatten to access as a 1D list

        # First subplot: Plotting residuals (True - Predicted)
        residuals = y_test - y_test_fit  # Calculate residuals
        residuals_train = y_train -  y_train_fit
        sample_indices = np.arange(len(y_test))  # X-axis as sample index
        sample_indices_train = np.arange(len(y_train))  # X-axis as sample index

        axes[0].scatter(sample_indices_train, residuals_train, color='blue', alpha=0.7, label="Residuals")
        axes[0].axhline(y=0, color='red', linestyle='--', label="Zero Residual")  # Reference line at 0
        axes[0].set_xlabel('Sample index')
        axes[0].set_ylabel('Residuals Train')
        axes[0].set_title(f'{CV_suffix}: R²: {r2_train:.2f}, MSE: {mse_train:.2f}, RMSE: {rmse_train:.2f}')
        axes[0].legend()

        # Second subplot: Plotting real vs predicted values based on their indices
        axes[1].plot(sample_indices_train, y_train, label='True Values', color='blue', linestyle='-', marker='o')
        axes[1].plot(sample_indices_train, y_train_fit, label='Predicted Values', color='red', linestyle='-', marker='x')
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Values')
        axes[1].set_title('Train: True vs Predicted')
        axes[1].legend()

        for j in range(residuals.shape[0]):
            axes[2].scatter(sample_indices, residuals[j, :], color='blue', alpha=0.7, label="Residuals")
            axes[2].axhline(y=0, color='red', linestyle='--', label="Zero Residual")  # Reference line at 0
            axes[2].set_xlabel('Sample index')
            axes[2].set_ylabel('Residuals Test')
            axes[2].set_title(f'{CV_suffix}: R²: {r2_test:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}')
            # axes[2].legend()
        handles, labels = axes[2].get_legend_handles_labels()
        fig.legend(handles, labels, loc = 'lower right', fontsize = 10)

        # Second subplot: Plotting real vs predicted values based on their indices
        axes[3].plot(sample_indices, y_test, label='True Values', color='blue', linestyle='-', marker='o')
        axes[3].plot(sample_indices, y_test_fit, label='Predicted Values', color='red', linestyle='-', marker='x')
        axes[3].set_xlabel('Sample Index')
        axes[3].set_ylabel('Values')
        axes[3].set_title('Test: True vs Predicted')
        axes[3].legend()

        # Adjust layout to make sure there's enough space between the subplots
        plt.tight_layout()
        # Show the plot
        # plt.show()
        

    def run_SVR(X_train, y_train, X_test, y_test, class_names=[], CV_suffix = "", opt = None, time_window = None):

        if opt == True:
            print('SVR hyper parameter tuning')
    
            # Define the model (without specifying kernel yet)
            model = svm.SVR()
    
            # Define hyperparameters to search, including different kernels
            param_grid = {
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Try different kernels
                'C': [0.1, 1, 10, 100],  # Regularization parameter
                'gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Kernel coefficient
                'epsilon': [0.1, 0.2, 0.5],  # Epsilon in the epsilon-SVR
            }
        
            # Perform Grid Search with Cross-Validation
            n_splits = 3
    
            grid_search = sklearn.model_selection.GridSearchCV(estimator=model, param_grid=param_grid, cv=n_splits, n_jobs=-1)
        
            # Fit the model with the best parameters found
            X_train.reshape((-1, ))
            y_train = y_train.to_numpy().reshape((-1, ))
            grid_search.fit(X_train, y_train)  #, groups=train_groups
        
            # Get the best parameters and model
            best_params = grid_search.best_params_
            best_model = grid_search.best_estimator_
    
            # Print the best parameters
            # print(f"Best hyperparameters: {best_params}")
    
            #  View cross-validation fold results
            cv_results = grid_search.cv_results_
            # print("Fold-wise test results:")
            for i in range(3):  # Assuming 3-fold cross-validation
                fold_score = cv_results[f'split{i}_test_score'][grid_search.best_index_]
                # print(f"Fold {i+1} test score: {fold_score}")
    
            # Evaluate the model on test data
            Overall = best_model.score(X_test, y_test)
            # print(f"Test R^2 score: {Overall}")
    
            model = best_model
            # Fit the model using the training data
            # model.fit(X_train, y_train)
        
        y_test_fit      = model.predict(X_test)
        r2_test         = model.score(X_test, y_test)
        mse             = round(np.mean(np.square(y_test - y_test_fit)), 2)
        rmse            = round(np.sqrt(mse), 2)
    
        y_train_fit     = model.predict(X_train)
        r2_train        = model.score(X_train, y_train)
        mse_train       = round(np.mean(np.square(y_train - y_train_fit)), 2)
        rmse_train      = round(np.sqrt(mse_train), 2)
        test_results    = (y_test_fit, r2_test, mse, rmse)
        train_results   = (y_train_fit, r2_train, mse_train, rmse_train)
        
        # Call the plot function with the necessary parameters
        Title = f"SVR_{CV_suffix}_{time_window}"
        # RPEModel.plot_results(y_test, y_test_fit, y_train, y_train_fit, r2_test, mse, rmse, r2_train, mse_train, rmse_train, CV_suffix= Title)
    
        return test_results, train_results
    
    def leave_p_out(self, data, RPE_or):
        n_windows = self.n_windows
        participants = self.participants

        RPE_predicted = np.zeros((len(participants), n_windows * 6))
        RPE_measured = np.zeros((len(participants), 6 * n_windows))

        for i in range(len(participants)):
        #-------------------------------------------------------------------------------------------------------------------------
            # Preprocessing
        #-------------------------------------------------------------------------------------------------------------------------
            # Create the test participant for cross validation
            test = np.zeros((n_windows * 6, data.shape[1]))
            RPE_test = np.zeros((n_windows * 6, 1))
            for j in range (6):
                start = len(participants) * n_windows * j + i * n_windows 
                test[n_windows * j : n_windows * (j + 1)] = data[start :  start + n_windows]
                RPE_test[n_windows * j : n_windows * (j + 1)] = RPE_or.iloc[start :  start + n_windows]

            # Check if test set was extracted correctly
            '''if n_windows == 3 and i == 5:
                print(f"For participant {participants[i]} test set is: {test}")
                print(test)'''
            
            # Remove the test participant
            for j in range(n_windows):  
                index = [(len(participants) * n_windows) * k for k in [0, 1, 2, 3, 4, 5]]
                index = [x + (j + i * n_windows) for x in index]
                if j == 0:
                    dataset = data.drop(index)
                    RPE = RPE_or.drop(index)
                else:
                    dataset = dataset.drop(index)
                    RPE = RPE.drop(index)

            # Shuffle the dataset
            dataset, RPE = shuffle(dataset, RPE, random_state = None)
            # MinMax scaling on training e test set
            scaler = MinMaxScaler()
            dataset = pd.DataFrame(scaler.fit_transform(dataset.values), columns = dataset.columns)
            test = pd.DataFrame(scaler.transform(test))

            # Apply pca
            pca = PCA() 
            dataset = pca.fit_transform(dataset.values)
            test = pca.transform(test)

            #----------------------------------------------------------------------------------------------------------------------
                # Linear Regression
            #----------------------------------------------------------------------------------------------------------------------
            linear_regression = LinearRegression().fit(dataset, RPE)
            X = linear_regression.coef_
            # Save coefficients
            if i == 0:
                coeff = np.zeros((len(participants) * n_windows, dataset.shape[1]))
            coeff[i, :] = X
            predicted = linear_regression.predict(test)
            # Visualize the intercept to the data


            # predicted = np.round(predicted)

            # Save predicted and actual value for comparison
            RPE_predicted[i, :] = predicted.T
            RPE_measured[i, :] = RPE_test.T

            #------------------------------------------------------------------------------------------------------------------------
                # Support vector regression
            #------------------------------------------------------------------------------------------------------------------------
            # With the method defined in the class
            if i == 0:
                results_svr_test = np.zeros((len(participants), len(RPE_test) + 3))
                results_svr_train = np.zeros((len(participants), len(RPE) * 6 + 3))

            test_results, train_results = RPEModel.run_SVR(dataset, RPE, test, RPE_test, class_names=[], CV_suffix = "", opt = True, time_window = None)
            
            results_svr_test[i, 0 : len(RPE_test)] = test_results[0]
            results_svr_test[i, len(RPE_test)] = test_results[1]
            results_svr_test[i, len(RPE_test) + 1] = test_results[2]
            results_svr_test[i, len(RPE_test) + 2] = test_results[3]

            results_svr_train[i, 0 : len(RPE)] = train_results[0]
            results_svr_train[i, len(RPE)] = train_results[1]
            results_svr_train[i, len(RPE) + 1] = train_results[2]
            results_svr_train[i, len(RPE) + 2] = train_results[3]


            # With the sklearn function
            '''svr = svm.SVR()
            RPE = RPE.to_numpy()
            RPE = RPE.reshape((-1, ))
            svr.fit(dataset, RPE)
            RPE_predicted_svr[i, :] = svr.predict(test)'''

        return RPE_measured, RPE_predicted, results_svr_test, results_svr_train


    def visualize_results_scatter(self, RPE_measured, RPE_predicted, length_window:int):
        participants = self.participants

        # Scattered data points
        plt.figure()
        for i in range(len(participants)):
            x = participants[i]
            y_measured = RPE_measured[i, :]
            y_predicted = RPE_predicted[i, :]

            # Color the background
            a = 0.01
            light_blue = (173/255, 216/255, 230/255)
            plt.axhspan(9, 11, color = light_blue, alpha = a)
            yellow = (1, 1, 0)
            plt.axhspan(11, 12, color = yellow, alpha = a)
            green = (0, 1, 0)
            plt.axhspan(12, 13, color = green, alpha = a)
            cyan = (0, 1, 1)
            plt.axhspan(13, 14, color = cyan, alpha = a)
            orange = (1, 0.647, 0)
            plt.axhspan(14, 15, color = orange, alpha = a)
            red = (1, 0, 0)
            plt.axhspan(15, 16, color = red, alpha = a)
            black = (0, 0, 0)
            plt.axhspan(16, 17, color = black, alpha = a)


            # Plot the data points, color coded accoding to their value
            for y1, y2 in zip(y_measured, y_predicted):
                if y1 <= 10:
                    c = light_blue
                elif y1 == 11:
                    c = yellow
                elif y1 == 12:
                    c = green
                elif y1 == 13:
                    c = cyan
                elif y1 == 14:
                    c = orange
                elif y1 == 15:
                    c = red
                elif y1 > 15:
                    c = black

                # plt.scatter(x + 0.5, y1, marker = 'x', color = c)
                plt.scatter(x, y2, marker = 'o', color = c)

        #--------------------------------------------------------------------------------------------------------
            # Measure the difference between model prediction and actual distribution: 
        #--------------------------------------------------------------------------------------------------------
        # R^2 value. Expected values between 0 and 1
        r_squared = r2_score(RPE_measured.T, RPE_predicted.T)
        # print(f"The R^2 value for subject {participants[i]} is: {r_squared:.4f}")
        print(f"The total R^2 value is: {r_squared}")

        # MSE and RMSE 
        mse = round(np.mean(np.square(RPE_measured - RPE_predicted)), 2)
        print(f"MSE is: {mse}")
        rmse = round(np.sqrt(mse), 2)
        print(f"RMSE is: {rmse}")

        # Legend for color coding the RPE signal
        legend_labels = ['<11 (Very light)', '11 (light)', '12 (light)', '13 (slightly hard)', '14 (hard)', '15 (really hard)', '>15 (exhaustion)']
        legend_colors = [light_blue, yellow, green, cyan, orange, red, black]
        legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in legend_colors]
        plt.legend(legend_handles, legend_labels, loc='upper center', ncol = 4, fontsize=8)
        plt.title(f"{length_window} second long window")

        plt.xlabel("Participant ID")
        plt.ylabel("RPE values")
        plt.xticks(participants)

        return plt
    

    # Visualize results as a curve
    def visualize_results_plot(self, RPE_measured, RPE_predicted, n_windows, fig, axs, handle1, handle2):
        x_axis = np.linspace(0, n_windows * 6 - 1, len(RPE_measured))

        axs[handle1, handle2].scatter(x_axis, RPE_measured, color = (1, 0, 0), marker = 'x', s = 20, label = "Reported values")
        axs[handle1, handle2].scatter(x_axis, RPE_predicted, color = (0, 0, 1), marker = 'o', s = 20, label = "Predicted values")
        
        handles, labels = axs[handle1, handle2].get_legend_handles_labels()
        fig.legend(handles, labels, loc = 'lower right', fontsize = 10)

        fig.tight_layout()
            

