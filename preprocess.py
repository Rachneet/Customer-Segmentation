import numpy as np
import pandas as pd
import helper_functions as helper
import xgboost as xgb
import gc
import shap

from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt


def preprocess_data(df, mode='train', process_outliers=True, impute=False):
    """
    :param df: dataframe to be cleaned
    :param mode: process data for train or test
    :param process_outliers: bool value for outlier processing
    :param impute: bool value for imputing missing data
    :return: imputed_df: cleaned dataframe
    """
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)

    # drop redundant cols # removal helps
    #     redundant_cols = ['D19_KONSUMTYP_MAX']
    #     df.drop(redundant_cols, axis=1, inplace=True)
    cols_to_keep = ['NATIONALITAET_KZ']

    print("START: Dropping columns with nulls")
    null_df = helper.get_null_values_df(df)
    helper.drop_cols_with_nulls(df, null_df)
    print("FINISHED")

    if mode == 'train':
        print("START: Dropping columns with mostly similar values")
        # remove cols with mostly similar values (>90%)
        sim_val_cols = helper.get_similar_value_cols(df)
        cols_to_drop = [col for col in sim_val_cols if col not in cols_to_keep]
        df.drop(cols_to_drop, axis=1, inplace=True)
        df.drop('LNR', axis=1, inplace=True)
        print("FINISHED")

    print("START: Convert unknowns to NANs")
    # convert unknowns to NANs
    df = helper.unknown_to_nans(df)
    print("FINISHED")

    # keep
    print("START: Engineer features")
    helper.create_features_from_cameo(df)
    helper.create_youth_movement_features(df)
    #     helper.create_features_from_lebensphase(df, 'LP_LEBENSPHASE_FEIN')
    df = helper.create_family_feature(df)
    df = helper.create_status_feature(df)
    print("FINISHED")

    print("START: Encoding categorical data")
    #     categoricals = ['CJT_GESAMTTYP']  # don't keep
    #     cols_present = np.intersect1d(np.array(categoricals), df.columns)
    #     df = helper.ohe_columns(df, cols_present)
    cols_to_enc = ['OST_WEST_KZ', 'D19_LETZTER_KAUF_BRANCHE', 'CAMEO_DEU_2015']
    unk_tokens = ['', 'D19_UNBEKANNT', 'XX']
    helper.encode_categorical_columns(df, cols_to_enc, unk_tokens)
    print("FINISHED")

    print("START: Handle mixed dtypes")
    df['CAMEO_DEUG_2015'] = df['CAMEO_DEUG_2015'].apply(lambda x: helper.handle_mixed_dtypes(x))
    helper.handle_datetime_column(df, 'EINGEFUEGT_AM')
    print("FINISHED")

    print("START: Handle numeric columns")
    # keep
    helper.bin_column(df, 'ANZ_PERSONEN', [0, 1, 2, 3, 4, 5, df['ANZ_PERSONEN'].max() + 1], False,
                      include_lowest=True, right=False)

    # handle numeric columns
    # build feature mapper
    cols = ['attributes', 'description', 'value', 'meaning']
    dias_attributes = pd.read_excel('data/DIAS Attributes - Values 2017.xlsx', index_col=False, skiprows=1)
    dias_attributes.drop("Unnamed: 0", axis=1, inplace=True)
    dias_attributes.ffill(inplace=True)
    num_cols = dias_attributes[dias_attributes.Meaning.str.contains('numeric')]['Attribute'].values
    new_num_cols = [col for col in num_cols if col != 'ANZ_PERSONEN']
    skew_cols = helper.get_skewed_columns(df[new_num_cols])
    # log transform skewed variables
    df[skew_cols.index] = df[skew_cols.index].apply(np.log1p)
    print("FINISHED")

    if mode == 'train' and process_outliers:
        print("START: Detect and remove outliers")
        # detect and remove outliers
        df_clean = helper.detect_outliers(df, new_num_cols)
        del df
        print("FINISHED")
    else:
        df_clean = df.copy()
        del df

    # do not impute
    if impute:
        print("START: Impute nulls")
        # impute nulls
        # impute numeric cols with median
        imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
        df_clean.loc[:, num_cols] = imp_median.fit_transform(df_clean[num_cols])
        # other columns are mostly categorical, we will replae NANs with most frequent values
        imp_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imputed_df = pd.DataFrame(imp_frequent.fit_transform(df_clean))
        imputed_df.columns = df_clean.columns
        del df_clean
        print("FINISHED")
    else:
        imputed_df = df_clean.copy()
        del df_clean

    gc.collect()

    return imputed_df


def get_feature_importance(X, y, params, plot=True):
    """
    :param X: array of training features
    :param y: array of target variable
    :param params: dict of model parameters
    :param plot: boolean value for plotting the chart
    :return: features_df: dataframe with feature importance values
    """
    model = xgb.XGBClassifier(**params)
    model.fit(X, y)

    ft_weights_xgb_reg = pd.DataFrame(model.feature_importances_, columns=['weight'],
                                      index=X.columns)

    if plot:
        ft_weights_xgb_reg.sort_values('weight', inplace=True)
        # Plotting feature importances
        plt.figure(figsize=(8, 20))
        plt.barh(ft_weights_xgb_reg.index, ft_weights_xgb_reg.weight, align='center')
        plt.title("Feature importances in the XGBoost model", fontsize=14)
        plt.xlabel("Feature importance")
        plt.margins(y=0.01)
        plt.show()

    features_df = ft_weights_xgb_reg.sort_values('weight', ascending=False)
    # features = ft_weights_xgb_reg[ft_weights_xgb_reg['weight'] > 0].index
    return features_df


def get_shapely_features(X, y, params):
    """
    :param X: array of training features
    :param y: array of target variable
    :param params: dict of model parameters
    :return:
    """

    model = xgb.XGBClassifier(**params)
    model.fit(X, y)
    # compute the SHAP values for every prediction in the test dataset
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    expected_values = explainer.expected_value

    # shap.summary_plot(shap_values, X_val, show=True, plot_type="bar")
    vals = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(X.columns, vals)), columns=['col_name', 'feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)

    return feature_importance


