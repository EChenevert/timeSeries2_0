import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.utils import shuffle
import random
import statsmodels.api as sm
from sklearn import linear_model
import textwrap


def outlierrm(df, thres=3):
    """Dont put in long lats in here! Need Year and Site name lol"""
    df = df.dropna()  #.set_index(['Simple site', 'level_1'])
    # print(df.columns.values, len(df.columns.values), len(df))
    # switch = False
    # if 'Basins' in df.columns.values or 'Community' in df.columns.values:
    #     print('True')
    #     switch = True
    #     holdstrings = df[['Basins', 'Community']]
    #     df = df.drop(['Basins', 'Community'], axis=1)
    df = df.apply(pd.to_numeric)
    length = len(df.columns.values)
    for col in df.columns.values:
        df[col + "_z"] = stats.zscore(df[col])
    for col in df.columns.values[length:]:
        df = df[np.abs(df[col]) < thres]
    # print('length 1: ', len(df))
    df = df.drop(df.columns.values[length:], axis=1)
    # print('length 2: ', len(df))
    # if switch:
    #     df = pd.concat([df, holdstrings], join='inner', axis=1)
    return df


def wrap_labels(ax, width, break_long_words=False):
    """
    https://medium.com/dunder-data/automatically-wrap-graph-labels-in-matplotlib-and-seaborn-a48740bc9ce
    :param ax:
    :param width:
    :param break_long_words:
    :return:
    """
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)


# https://www.analyticsvidhya.com/blog/2020/10/a-comprehensive-guide-to-feature-selection-using-wrapper-methods-in-python/#:~:text=1.-,Forward%20selection,with%20all%20other%20remaining%20features.
def backward_elimination(data, target, num_feats=5, significance_level=0.05):
    target = list(target)
    features = data.columns.tolist()
    while(len(features)>0):
        features_with_constant = sm.add_constant(data[features])
        p_values = sm.OLS(target, features_with_constant).fit().pvalues[1:]
        max_p_value = p_values.max()
        if(max_p_value >= significance_level) or (len(features) > num_feats):
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break
    return features


def unscaled_weights_from_full_standardization(X, y, bayesianReg: linear_model):
    """
    https://stackoverflow.com/questions/57513372/can-i-inverse-transform-the-intercept-and-coefficients-of-
    lasso-regression-after
    Better source:
    https://stats.stackexchange.com/questions/74622/converting-standardized-betas-back-to-original-variables
    """
    a = bayesianReg.coef_

    # Me tryna do my own thing
    coefs_new = []
    for x in range(len(X.columns)):
        # print(X.columns.values[x])
        col = X.columns.values[x]
        coefs_new.append(((a[x] * float(y.std())) / (np.asarray(X.std()[col]))))
    intercept = float(y.std()) - np.sum(np.multiply(np.asarray(coefs_new), np.asarray(X.mean())))  # hadamard product

    return coefs_new, intercept

def unscaled_weights_from_Xstandardized(X, bayesianReg: linear_model):
    """
    https://stackoverflow.com/questions/57513372/can-i-inverse-transform-the-intercept-and-coefficients-of-
    lasso-regression-after
    Better source:
    https://stats.stackexchange.com/questions/74622/converting-standardized-betas-back-to-original-variables
    """
    a = bayesianReg.coef_
    i = bayesianReg.intercept_
    # Me tryna do my own thing
    coefs_new = []
    for x in range(len(X.columns)):
        # print(X.columns.values[x])
        col = X.columns.values[x]
        coefs_new.append((a[x] / (np.asarray(X.std()[col]))))
    intercept = i - np.sum(np.multiply(np.asarray(coefs_new), np.asarray(X.mean())))  # hadamard product

    return coefs_new, intercept

def ln_transform_weights(coefs):
    exp_coefs = np.exp(coefs)
    # print(" exponential ", exp_coefs)
    minus_coefs = exp_coefs - 1
    # minus_coefs = [i - 1 for i in exp_coefs]
    # print("Minus 1: ", minus_coefs)
    new_coefs = minus_coefs * 100
    # print(" multiply 100: ", new_coefs)
    return new_coefs

def log10_transform_weights(coefs):
    """ Coefs must be a list"""
    # exp_coefs = [10 ** w_i for w_i in coefs]
    coefs = np.asarray(coefs)
    exp_coefs = 10 ** coefs
    minus_coefs = exp_coefs - 1
    new_coefs = minus_coefs * 100
    return list(new_coefs)

# log_transform_weights(unscaled_weights)


def log10_cv_results_and_plot(bay_model, bestfeatures, unscaled_predictor_matrix, predictor_matrix, target,
                        color_scheme: dict, marsh_key):
    # Error Containers
    predicted = []  # holds they predicted values of y
    y_ls = []  # holds the true values of y
    residuals = []

    # Performance Metric Containers: I allow use the median because I want to be more robust to outliers
    r2_total_medians = []  # holds the k-fold median r^2 value. Will be length of 100 due to 100 repeats
    mae_total_medians = []  # holds the k-fold median Mean Absolute Error (MAE) value. Will be length of 100 due to 100 repeats

    # parameter holders
    weight_vector_ls = []  # holds the learned parameters for each k-fold test
    regularizor_ls = []  # holds the learned L2 regularization term for each k-fold test
    unscaled_w_ls = []  # holds the inverted weights to their natural scales
    intercept_ls = []  # holds the inverted intercept to its natural scale
    weight_certainty_ls = []  # holds the number of well-determinned parameters for each k-fold test
    prediction_certainty_ls = []  # holds the standard deviations of the predictions (predictive distributions)
    prediction_list = []

    for i in range(200):  # for 100 repeats
        try_cv = KFold(n_splits=5, shuffle=True)

        # Scaled lists
        r2_ls = []
        mae_ls = []

        # Certainty lists
        pred_certain = []
        pred_list = []
        w_certain = []

        for train_index, test_index in try_cv.split(predictor_matrix):
            X_train, X_test = predictor_matrix.iloc[train_index], predictor_matrix.iloc[test_index]
            y_train, y_test = target.iloc[train_index], target.iloc[test_index]
            # Fit the model
            bay_model.fit(X_train, y_train.values.ravel())
            # collect unscaled parameters
            unscaled_weights, intercept = unscaled_weights_from_Xstandardized(unscaled_predictor_matrix[bestfeatures],
                                                                              bay_model)
            # Log10 transform the weights (since log10 is used on dependet variable)
            unscaled_transformed_weights = log10_transform_weights(unscaled_weights)
            # save
            unscaled_w_ls.append(unscaled_transformed_weights)

            intercept_ls.append(intercept)
            # Collect scaled parameters
            weights = bay_model.coef_
            weight_vector_ls.append(abs(weights))  # Take the absolute values of weights for relative feature importance
            regularizor = bay_model.lambda_ / bay_model.alpha_
            regularizor_ls.append(regularizor)
            design_m = np.asarray(X_train)
            eigs = np.linalg.eigh(bay_model.lambda_ * (design_m.T @ design_m))
            weight_certainty = []
            for eig in eigs[0]:
                weight_certainty.append(eig / (eig + bay_model.lambda_))
            weight_certainty = np.sum(weight_certainty)
            w_certain.append(weight_certainty)
            # Make our predictions for y
            ypred, ystd = bay_model.predict(X_test, return_std=True)
            # Save average std on each prediction
            #         pred_certain.append(np.mean(ystd))

            pred_list += list(10 ** ypred)
            pred_certain += list(10 ** ystd)

            # Metrics for scaled y: ESSENTIAL
            exp10_y_test = 10 ** y_test
            exp10_ypred = 10 ** ypred
            r2 = r2_score(exp10_y_test, exp10_ypred)
            r2_ls.append(r2)
            mae = mean_absolute_error(exp10_y_test, exp10_ypred)
            mae_ls.append(mae)


        # Average certainty in predictions
        prediction_certainty_ls.append(np.mean(pred_certain))
        prediction_list.append(pred_list)

        weight_certainty_ls.append(np.mean(w_certain))
        # Average predictions over the Kfold first: scaled
        r2_median = np.median(r2_ls)
        r2_total_medians.append(r2_median)
        mae_median = np.median(mae_ls)
        mae_total_medians.append(mae_median)

        predicted = predicted + list(cross_val_predict(bay_model, predictor_matrix, target.values.ravel(), cv=try_cv))
        residuals = residuals + list(target.values.ravel() - cross_val_predict(bay_model, predictor_matrix,
                                                                               target.values.ravel(), cv=try_cv))
        y_ls += list(target.values.ravel())

    # Add each of the model parameters to a dictionary
    weight_df = pd.DataFrame(weight_vector_ls, columns=bestfeatures)
    unscaled_weight_df = pd.DataFrame(unscaled_w_ls, columns=bestfeatures)

    # Now calculate the mean of th kfold means for each repeat: scaled accretion
    r2_final_median = np.median(r2_total_medians)
    mae_final_median = np.median(mae_total_medians)

    exp10_y_ls = [10 ** y_i for y_i in y_ls]
    exp10_predicted = [10 ** y_i for y_i in predicted]

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(9, 8))
    hb = ax.hexbin(x=exp10_y_ls,
                   y=exp10_predicted,
                   gridsize=30, edgecolors='grey',
                   cmap=color_scheme['cmap'], mincnt=1)
    ax.set_facecolor('white')
    ax.set_xlabel("Measured Accretion Rate (mm/yr)")
    ax.set_ylabel("Estimated Accretion Rate (mm/yr)")
    ax.set_title(marsh_key + " Sites")
    cb = fig.colorbar(hb, ax=ax)
    cb.ax.get_yaxis().labelpad = 20
    cb.set_label('Density of Predictions', rotation=270)

    exp10_y = 10 ** target

    ax.plot([exp10_y.min(), exp10_y.max()], [exp10_y.min(), exp10_y.max()],
            color_scheme['line'], lw=3)

    ax.annotate("Median r-squared = {:.3f}".format(r2_final_median), xy=(20, 410), xycoords='axes points',
                bbox=dict(boxstyle='round', fc='w'),
                size=15, ha='left', va='top')
    ax.annotate("Median MAE = {:.3f}".format(mae_final_median), xy=(20, 380), xycoords='axes points',
                bbox=dict(boxstyle='round', fc='w'),
                size=15, ha='left', va='top')
    fig.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\scaled_X_LOG\\" + marsh_key +
                "_scaledX_nolog_cv_human.eps", format='eps',
                dpi=300,
                bbox_inches='tight')
    plt.show()


    # save all results in a dictionary
    dictionary = {
        "Scaled Weights": weight_df, "Unscaled Weights": unscaled_weight_df, "Unscaled Intercepts": intercept_ls,
        "Scaled regularizors": regularizor_ls, "# Well Determined Weights": weight_certainty_ls,
        "Standard Deviations of Predictions": prediction_certainty_ls, "Predictions": prediction_list,
        "Residuals": residuals, "Predicted for Residuals": predicted
    }

    # lets just look at the residuals.... why not???
    fig, ax = plt.subplots(figsize=(9, 7))
    hb = ax.hexbin(x=dictionary['Predicted for Residuals'],
                   y=dictionary['Residuals'],
                   gridsize=30, edgecolors='grey',
                   cmap='YlGnBu', mincnt=1)
    ax.set_facecolor('white')
    ax.set_xlabel("Fitted Value (Prediction)")
    ax.set_ylabel("Residual (y_true - y_predicted)")
    ax.set_title(marsh_key)
    cb = fig.colorbar(hb, ax=ax)
    cb.ax.get_yaxis().labelpad = 15
    cb.set_label('Density of Residuals', rotation=270)
    ax.axhline(0.0, linestyle='--')
    plt.show()

    return dictionary











# Below is function if I do not rescale anything, for sake of not making a mistake


def log10_cv_results_and_plot2(bay_model, bestfeatures, unscaled_predictor_matrix, predictor_matrix, target,
                        color_scheme: dict, marsh_key):
    # Error Containers
    predicted = []  # holds they predicted values of y
    y_ls = []  # holds the true values of y
    residuals = []

    # Performance Metric Containers: I allow use the median because I want to be more robust to outliers
    r2_total_medians = []  # holds the k-fold median r^2 value. Will be length of 100 due to 100 repeats
    mae_total_medians = []  # holds the k-fold median Mean Absolute Error (MAE) value. Will be length of 100 due to 100 repeats

    # parameter holders
    weight_vector_ls = []  # holds the learned parameters for each k-fold test
    regularizor_ls = []  # holds the learned L2 regularization term for each k-fold test
    unscaled_w_ls = []  # holds the inverted weights to their natural scales
    intercept_ls = []  # holds the inverted intercept to its natural scale
    weight_certainty_ls = []  # holds the number of well-determinned parameters for each k-fold test
    prediction_certainty_ls = []  # holds the standard deviations of the predictions (predictive distributions)
    prediction_list = []

    for i in range(200):  # for 100 repeats
        try_cv = KFold(n_splits=5, shuffle=True)

        # Scaled lists
        r2_ls = []
        mae_ls = []

        # Certainty lists
        pred_certain = []
        pred_list = []
        w_certain = []

        for train_index, test_index in try_cv.split(predictor_matrix):
            X_train, X_test = predictor_matrix.iloc[train_index], predictor_matrix.iloc[test_index]
            y_train, y_test = target.iloc[train_index], target.iloc[test_index]
            # Fit the model
            bay_model.fit(X_train, y_train.values.ravel())
            # # collect unscaled parameters
            # unscaled_weights, intercept = unscaled_weights_from_Xstandardized(unscaled_predictor_matrix[bestfeatures],
            #                                                                   bay_model)
            # # Log10 transform the weights (since log10 is used on dependet variable)
            # unscaled_transformed_weights = log10_transform_weights(unscaled_weights)
            # save
            unscaled_w_ls.append(bay_model.coef_)

            intercept_ls.append(bay_model.intercept_)
            # Collect scaled parameters
            weights = bay_model.coef_
            weight_vector_ls.append(abs(weights))  # Take the absolute values of weights for relative feature importance
            regularizor = bay_model.lambda_ / bay_model.alpha_
            regularizor_ls.append(regularizor)
            design_m = np.asarray(X_train)
            eigs = np.linalg.eigh(bay_model.lambda_ * (design_m.T @ design_m))
            weight_certainty = []
            for eig in eigs[0]:
                weight_certainty.append(eig / (eig + bay_model.lambda_))
            weight_certainty = np.sum(weight_certainty)
            w_certain.append(weight_certainty)
            # Make our predictions for y
            ypred, ystd = bay_model.predict(X_test, return_std=True)
            # Save average std on each prediction
            #         pred_certain.append(np.mean(ystd))

            pred_list += list(10 ** ypred)
            pred_certain += list(10 ** ystd)

            # Metrics for scaled y: ESSENTIAL
            exp10_y_test = 10 ** y_test
            exp10_ypred = 10 ** ypred
            r2 = r2_score(exp10_y_test, exp10_ypred)
            r2_ls.append(r2)
            mae = mean_absolute_error(exp10_y_test, exp10_ypred)
            mae_ls.append(mae)


        # Average certainty in predictions
        prediction_certainty_ls.append(np.mean(pred_certain))
        prediction_list.append(pred_list)

        weight_certainty_ls.append(np.mean(w_certain))
        # Average predictions over the Kfold first: scaled
        r2_median = np.median(r2_ls)
        r2_total_medians.append(r2_median)
        mae_median = np.median(mae_ls)
        mae_total_medians.append(mae_median)

        predicted = predicted + list(cross_val_predict(bay_model, predictor_matrix, target.values.ravel(), cv=try_cv))
        residuals = residuals + list(target.values.ravel() - cross_val_predict(bay_model, predictor_matrix,
                                                                               target.values.ravel(), cv=try_cv))
        y_ls += list(target.values.ravel())

    # Add each of the model parameters to a dictionary
    weight_df = pd.DataFrame(weight_vector_ls, columns=bestfeatures)
    unscaled_weight_df = pd.DataFrame(unscaled_w_ls, columns=bestfeatures)

    # Now calculate the mean of th kfold means for each repeat: scaled accretion
    r2_final_median = np.median(r2_total_medians)
    mae_final_median = np.median(mae_total_medians)

    exp10_y_ls = [10 ** y_i for y_i in y_ls]
    exp10_predicted = [10 ** y_i for y_i in predicted]

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(9, 8))
    hb = ax.hexbin(x=exp10_y_ls,
                   y=exp10_predicted,
                   gridsize=30, edgecolors='grey',
                   cmap=color_scheme['cmap'], mincnt=1)
    ax.set_facecolor('white')
    ax.set_xlabel("Measured Accretion Rate (mm/yr)")
    ax.set_ylabel("Estimated Accretion Rate (mm/yr)")
    ax.set_title(marsh_key + " Sites")
    cb = fig.colorbar(hb, ax=ax)
    cb.ax.get_yaxis().labelpad = 20
    cb.set_label('Density of Predictions', rotation=270)

    exp10_y = 10 ** target

    ax.plot([exp10_y.min(), exp10_y.max()], [exp10_y.min(), exp10_y.max()],
            color_scheme['line'], lw=3)

    ax.annotate("Median r-squared = {:.3f}".format(r2_final_median), xy=(20, 410), xycoords='axes points',
                bbox=dict(boxstyle='round', fc='w'),
                size=15, ha='left', va='top')
    ax.annotate("Median MAE = {:.3f}".format(mae_final_median), xy=(20, 380), xycoords='axes points',
                bbox=dict(boxstyle='round', fc='w'),
                size=15, ha='left', va='top')
    fig.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\scaled_X_LOG\\" + marsh_key +
                "_scaledX_nolog_cv_human.eps", format='eps',
                dpi=300,
                bbox_inches='tight')
    plt.show()


    # save all results in a dictionary
    dictionary = {
        "Scaled Weights": weight_df, "Unscaled Weights": unscaled_weight_df, "Unscaled Intercepts": intercept_ls,
        "Scaled regularizors": regularizor_ls, "# Well Determined Weights": weight_certainty_ls,
        "Standard Deviations of Predictions": prediction_certainty_ls, "Predictions": prediction_list,
        "Residuals": residuals, "Predicted for Residuals": predicted
    }

    # lets just look at the residuals.... why not???
    fig, ax = plt.subplots(figsize=(9, 7))
    hb = ax.hexbin(x=dictionary['Predicted for Residuals'],
                   y=dictionary['Residuals'],
                   gridsize=30, edgecolors='grey',
                   cmap='YlGnBu', mincnt=1)
    ax.set_facecolor('white')
    ax.set_xlabel("Fitted Value (Prediction)")
    ax.set_ylabel("Residual (y_true - y_predicted)")
    ax.set_title(marsh_key)
    cb = fig.colorbar(hb, ax=ax)
    cb.ax.get_yaxis().labelpad = 15
    cb.set_label('Density of Residuals', rotation=270)
    ax.axhline(0.0, linestyle='--')
    plt.show()

    return dictionary




def cv_results_and_plot(bay_model, bestfeatures, unscaled_predictor_matrix, predictor_matrix, target,
                        color_scheme: dict, marsh_key):
    # Error Containers
    predicted = []  # holds they predicted values of y
    y_ls = []  # holds the true values of y
    residuals = []

    # Performance Metric Containers: I allow use the median because I want to be more robust to outliers
    r2_total_medians = []  # holds the k-fold median r^2 value. Will be length of 100 due to 100 repeats
    mae_total_medians = []  # holds the k-fold median Mean Absolute Error (MAE) value. Will be length of 100 due to 100 repeats

    # parameter holders
    weight_vector_ls = []  # holds the learned parameters for each k-fold test
    regularizor_ls = []  # holds the learned L2 regularization term for each k-fold test
    unscaled_w_ls = []  # holds the inverted weights to their natural scales
    intercept_ls = []  # holds the inverted intercept to its natural scale
    weight_certainty_ls = []  # holds the number of well-determinned parameters for each k-fold test
    prediction_certainty_ls = []  # holds the standard deviations of the predictions (predictive distributions)
    prediction_list = []

    for i in range(200):  # for 100 repeats
        try_cv = KFold(n_splits=5, shuffle=True)

        # Scaled lists
        r2_ls = []
        mae_ls = []

        # Certainty lists
        pred_certain = []
        pred_list = []
        w_certain = []

        for train_index, test_index in try_cv.split(predictor_matrix):
            X_train, X_test = predictor_matrix.iloc[train_index], predictor_matrix.iloc[test_index]
            y_train, y_test = target.iloc[train_index], target.iloc[test_index]
            # Fit the model
            bay_model.fit(X_train, y_train.values.ravel())
            # collect unscaled parameters
            unscaled_weights, intercept = unscaled_weights_from_Xstandardized(unscaled_predictor_matrix[bestfeatures],
                                                                              bay_model)
            # # Log10 transform the weights (since log10 is used on dependent variable)
            # unscaled_transformed_weights = log10_transform_weights(unscaled_weights)
            # # save
            unscaled_w_ls.append(unscaled_weights)

            intercept_ls.append(intercept)
            # Collect scaled parameters
            weights = bay_model.coef_
            weight_vector_ls.append(abs(weights))  # Take the absolute values of weights for relative feature importance
            regularizor = bay_model.lambda_ / bay_model.alpha_
            regularizor_ls.append(regularizor)
            design_m = np.asarray(X_train)
            eigs = np.linalg.eigh(bay_model.lambda_ * (design_m.T @ design_m))
            weight_certainty = []
            for eig in eigs[0]:
                weight_certainty.append(eig / (eig + bay_model.lambda_))
            weight_certainty = np.sum(weight_certainty)
            w_certain.append(weight_certainty)
            # Make our predictions for y
            ypred, ystd = bay_model.predict(X_test, return_std=True)
            # Save average std on each prediction
            #         pred_certain.append(np.mean(ystd))

            pred_list += list(ypred)
            pred_certain += list(ystd)

            # Metrics for scaled y: ESSENTIAL
            r2 = r2_score(y_test, ypred)
            r2_ls.append(r2)
            mae = mean_absolute_error(y_test, ypred)
            mae_ls.append(mae)


        # Average certainty in predictions
        prediction_certainty_ls.append(np.mean(pred_certain))
        prediction_list.append(pred_list)

        weight_certainty_ls.append(np.mean(w_certain))
        # Average predictions over the Kfold first: scaled
        r2_median = np.median(r2_ls)
        r2_total_medians.append(r2_median)
        mae_median = np.median(mae_ls)
        mae_total_medians.append(mae_median)

        predicted = predicted + list(cross_val_predict(bay_model, predictor_matrix, target.values.ravel(), cv=try_cv))
        residuals = residuals + list(target.values.ravel() - cross_val_predict(bay_model, predictor_matrix,
                                                                               target.values.ravel(), cv=try_cv))
        y_ls += list(target.values.ravel())

    # Add each of the model parameters to a dictionary
    weight_df = pd.DataFrame(weight_vector_ls, columns=bestfeatures)
    unscaled_weight_df = pd.DataFrame(unscaled_w_ls, columns=bestfeatures)

    # Now calculate the mean of th kfold means for each repeat: scaled accretion
    r2_final_median = np.median(r2_total_medians)
    mae_final_median = np.median(mae_total_medians)

    # exp10_y_ls = [10 ** y_i for y_i in y_ls]
    # exp10_predicted = [10 ** y_i for y_i in predicted]

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(9, 8))
    hb = ax.hexbin(x=y_ls,
                   y=predicted,
                   gridsize=30, edgecolors='grey',
                   cmap=color_scheme['cmap'], mincnt=1)
    ax.set_facecolor('white')
    ax.set_xlabel("Measured Accretion Rate (mm/yr)")
    ax.set_ylabel("Estimated Accretion Rate (mm/yr)")
    ax.set_title(marsh_key + " Sites")
    cb = fig.colorbar(hb, ax=ax)
    cb.ax.get_yaxis().labelpad = 20
    cb.set_label('Density of Predictions', rotation=270)

    # exp10_y = 10 ** target

    ax.plot([target.min(), target.max()], [target.min(), target.max()],
            color_scheme['line'], lw=3)

    ax.annotate("Median r-squared = {:.3f}".format(r2_final_median), xy=(20, 410), xycoords='axes points',
                bbox=dict(boxstyle='round', fc='w'),
                size=15, ha='left', va='top')
    ax.annotate("Median MAE = {:.3f}".format(mae_final_median), xy=(20, 380), xycoords='axes points',
                bbox=dict(boxstyle='round', fc='w'),
                size=15, ha='left', va='top')
    # fig.savefig("D:\\Etienne\\fall2022\\agu_data\\results\\scaled_X_LOG\\" + marsh_key +
    #             "_scaledX_nolog_cv_human.eps", format='eps',
    #             dpi=300,
    #             bbox_inches='tight')
    plt.show()


    # save all results in a dictionary
    dictionary = {
        "Scaled Weights": weight_df, "Unscaled Weights": unscaled_weight_df, "Unscaled Intercepts": intercept_ls,
        "Scaled regularizors": regularizor_ls, "# Well Determined Weights": weight_certainty_ls,
        "Standard Deviations of Predictions": prediction_certainty_ls, "Predictions": prediction_list,
        "Residuals": residuals, "Predicted for Residuals": predicted
    }

    # lets just look at the residuals.... why not???
    fig, ax = plt.subplots(figsize=(9, 7))
    hb = ax.hexbin(x=dictionary['Predicted for Residuals'],
                   y=dictionary['Residuals'],
                   gridsize=30, edgecolors='grey',
                   cmap='YlGnBu', mincnt=1)
    ax.set_facecolor('white')
    ax.set_xlabel("Fitted Value (Prediction)")
    ax.set_ylabel("Residual (y_true - y_predicted)")
    ax.set_title(marsh_key)
    cb = fig.colorbar(hb, ax=ax)
    cb.ax.get_yaxis().labelpad = 15
    cb.set_label('Density of Residuals', rotation=270)
    ax.axhline(0.0, linestyle='--')
    plt.show()

    return dictionary


