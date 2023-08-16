import os
import sys

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import permutation_importance
from sklearn.utils.validation import check_consistent_length, check_array
from xgboost import XGBRegressor
import matplotlib.pyplot as plt


def grangers_causation_matrix(data, variables, max_lag=12, test="ssr_chi2test"):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table
    are the P-Values. P-Values lesser than the significance level (0.05), implies
    the Null Hypothesis that the coefficients of the corresponding past values is
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            if "cell" in c and "cell" in r:
                continue
            elif "rand" in c and "rand" in r:
                continue
            print(r, c)
            test_result = grangercausalitytests(data[[r, c]], maxlag=max_lag, verbose=False)
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(max_lag)]
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + "_x" for var in variables]
    df.index = [var + "_y" for var in variables]
    return df


def cointegration_test(df, alpha=0.05):
    """Perform Johanson"s Cointegration Test and Report Summary"""
    out = coint_johansen(df, -1, 5)
    d = {"0.90": 0, "0.95": 1, "0.99": 2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1 - alpha)]]

    def adjust(val, length=6):
        return str(val).ljust(length)

    # Summary
    print("Name   ::  Test Stat > C(95%)    =>   Signif  \n", "--" * 20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ":: ", adjust(round(trace, 2), 9), ">", adjust(cvt, 8), " =>  ", trace > cvt)


def adfuller_test(series, signif=0.05, name=""):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag="AIC")
    output = {"test_statistic": round(r[0], 4), "pvalue": round(r[1], 4), "n_lags": round(r[2], 4), "n_obs": r[3]}
    p_value = output["pvalue"]

    def adjust(val, length=6):
        return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}": ')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")


def fit_model(df):
    model = VAR(df)
    result = model.fit(4)
    print("AIC : ", result.aic)
    print("BIC : ", result.bic)
    print("FPE : ", result.fpe)
    print("HQIC: ", result.hqic)
    return result


def residuals(df, model_fitted):
    out = durbin_watson(model_fitted.resid)
    for col, val in zip(df.columns, out):
        print(col, ":", round(val, 2))


def mean_absolute_percentage_error(y_true, y_pred,
                                   sample_weight=None,
                                   multioutput='uniform_average'):
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)
    epsilon = np.finfo(np.float64).eps
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    output_errors = np.average(mape,
                               weights=sample_weight, axis=0)
    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def _check_reg_targets(y_true, y_pred, multioutput, dtype="numeric"):
    """Check that y_true and y_pred belong to the same regression task.
    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    multioutput : array-like or string in ['raw_values', uniform_average',
        'variance_weighted'] or None
        None is accepted due to backward compatibility of r2_score().
    Returns
    -------
    type_true : one of {'continuous', continuous-multioutput'}
        The type of the true target data, as output by
        'utils.multiclass.type_of_target'.
    y_true : array-like of shape (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples, n_outputs)
        Estimated target values.
    multioutput : array-like of shape (n_outputs) or string in ['raw_values',
        uniform_average', 'variance_weighted'] or None
        Custom output weights if ``multioutput`` is array-like or
        just the corresponding argument if ``multioutput`` is a
        correct keyword.
    dtype : str or list, default="numeric"
        the dtype argument passed to check_array.
    """
    check_consistent_length(y_true, y_pred)
    y_true = check_array(y_true, ensure_2d=False, dtype=dtype)
    y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError("y_true and y_pred have different number of output "
                         "({0}!={1})".format(y_true.shape[1], y_pred.shape[1]))

    n_outputs = y_true.shape[1]
    allowed_multioutput_str = ('raw_values', 'uniform_average',
                               'variance_weighted')
    if isinstance(multioutput, str):
        if multioutput not in allowed_multioutput_str:
            raise ValueError("Allowed 'multioutput' string values are {}. "
                             "You provided multioutput={!r}".format(
                                 allowed_multioutput_str,
                                 multioutput))
    elif multioutput is not None:
        multioutput = check_array(multioutput, ensure_2d=False)
        if n_outputs == 1:
            raise ValueError("Custom weights are useful only in "
                             "multi-output cases.")
        elif n_outputs != len(multioutput):
            raise ValueError(("There must be equally many custom weights "
                              "(%d) as outputs (%d).") %
                             (len(multioutput), n_outputs))
    y_type = 'continuous' if n_outputs == 1 else 'continuous-multioutput'

    return y_type, y_true, y_pred, multioutput


def fit_predict(df, model_name, features, n_splits):
    split = TimeSeriesSplit(n_splits=n_splits)
    models, mae, mse, mape = [], [], [], []
    for train_index, test_index in split.split(df):
        cv_train, cv_test = df.iloc[train_index], df.iloc[test_index]
        x_train, x_test, y_train, y_test = cv_train[features].values, cv_test[features].values, \
                                           cv_train["n_spikes"].values, cv_test["n_spikes"].values
        model = eval(model_name)
        model.fit(x_train, y_train)  # , sample_weight=np.exp(y_train - 1))
        y_predicted = model.predict(x_test)
        mae.append(mean_absolute_error(y_test, y_predicted))
        mse.append(mean_squared_error(y_test, y_predicted))
        mape.append(mean_absolute_percentage_error(y_test, y_predicted))
        models.append(model)
    print("The {} performance".format(model_name.split("(")[0]))
    print("MAE is [{0},{1},{2}]  MSE is [{3},{4},{5}]  MAPE is [{6},{7},{8}]".format(round(min(mae), 2),
                                                                                     round(np.median(mae), 2),
                                                                                     round(max(mae), 2),
                                                                                     round(min(mse), 2),
                                                                                     round(np.median(mse), 2),
                                                                                     round(max(mse), 2),
                                                                                     round(min(mape), 2),
                                                                                     round(np.median(mape), 2),
                                                                                     round(max(mape), 2)))
    print("--------------------------------------")
    return models, mae, mse, mape


class RandomRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, range):
        self._range = range

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.random.random(size=len(X)) * (self._range[1] - self._range[0]) - self._range[0]


def feature_importance(df, random_state=0, n_repeats=5, n_splits=5, out_dir="output"):
    # check hyperparams of the detector, hold-out validation
    # check multi-collinearity
    features = ["area", "perimeter", "orientation", "velocity_x", "velocity_y", "elongation"]#, "short", "long",
                #"solidity", "eccentricity", "n_cells"]
    x, y = df[features].values, df["n_spikes"].values
    models = ["RandomRegressor(range=({0}, {1}))".format(np.min(y), np.max(y)),
              "DummyRegressor()",
              "KNeighborsRegressor(n_jobs=-1)",
              "RandomForestRegressor(random_state={}, n_jobs=-1)".format(random_state),
              "XGBRegressor(random_state={}, n_jobs=-1)".format(random_state)]
    out = pd.DataFrame()
    for model_name in models:
        models, mae, mse, mape = fit_predict(df=df, model_name=model_name, features=features, n_splits=n_splits)
        best_model = models[np.argmin(mae)]
        best_model.fit(x, y)  # , sample_weight=np.exp(y - 1))
        importance = permutation_importance(best_model, x, y, scoring="neg_mean_absolute_percentage_error",
                                            n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
        d = {"model": model_name.split("(")[0]}
        for i in range(n_splits):
            d["_".join(["mae", str(i)])] = mae[i]
            d["_".join(["mse", str(i)])] = mse[i]
            d["_".join(["mape", str(i)])] = mape[i]
        for i, f in enumerate(features):
            for j in range(n_repeats):
                d["_".join([f, str(j)])] = importance.importances[i, j]
        out = out.append(d, ignore_index=True)
    out.to_csv(os.path.join(out_dir, "{}.csv".format(random_state)), sep=",", index=False)


def read_data(directory):
    df = None
    for file in os.listdir(directory):
        if not file.endswith("csv"):
            continue
        d = pd.read_csv(os.path.join(directory, file), sep=",")
        d["seed"] = int(file.split(".")[0])
        if df is None:
            df = pd.DataFrame(columns=d.columns)
        df = df.append(d)
    df = df[df["model"] != "LinearRegression"]
    return df


def plot_vars(df):
    features = ["area", "perimeter", "short", "long", "orientation", "solidity", "elongation", "eccentricity",
                "n_cells"]
    fig, axes = plt.subplots(figsize=(20, 5), nrows=1, ncols=len(features))
    for var1, ax in zip(features, axes):
        ax.hist(df[var1])
        ax.set_xlabel(var1)
    plt.savefig("relationship.png")


if __name__ == "__main__":
    try:
        seed = int(sys.argv[1])
    except:
        seed = 0
    data = pd.read_csv("xenobits.csv", sep=",", index_col=0)
    data["n_spikes"] += 1
    for v in ["x", "y"]:
        var = "_".join(["velocity", v])
        data[var] = data[v].diff().rolling(50).mean()
        data.loc[0, var] = 0.0
    data.dropna(inplace=True)
    # data = data[(data["n_cells"] < 50) & (data["perimeter"] < 1000) & (data["short"] < 150) & (data["area"] < 20000) &
    #             (data["long"] < 100)]
    # process_xenobits_data(df=data)
    # plot_vars(df=data)
    feature_importance(df=data, random_state=seed)
    # data2 = pd.DataFrame(np.random.randint(0, 2, size=data.shape),
    #                      columns=[col.replace("cell", "rand") for col in data.columns])
    # print(data["cell_5"])
    # data.drop(["cell_5", "cell_7"], axis=1, inplace=True)
    # data = pd.concat([data, data2], axis=1)
    # data = data + 0.00001 * np.random.rand(*data.shape)
    # values = grangers_causation_matrix(data=data, variables=data.columns)
    # values.to_csv("grangers_mixed.csv", sep=",", index=False)
    # cointegration_test(data)
    # d = pd.read_csv("grangers.csv", sep=",")
    # d = d.iloc[:int(d.shape[1] / 2)]
    # print(d.shape)
    # c = 0
    # s = 0
    # for col in d.columns:
    #     c += len(d[d[col] < 0.05])
    #     s += len(d[col])
    # print(c / s)
    # data = data.diff().iloc[1:]
    # for name, column in data.iteritems():
    #     adfuller_test(column, name=column.name)
    # model = fit_model(df=data)
    # residuals(df=data, model_fitted=model)
    # print(fit_predict(df=data))
