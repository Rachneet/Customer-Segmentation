import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import itertools
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

import plotly
import plotly.graph_objects as go


def ohe_columns(df, columns):
    """
    :param df: input data in the form of a dataframe
    :param columns: list of columns to encode
    :return: df: dataframe with the encoded columns
    """
    for col in columns:
        df = pd.concat([df.drop([col], axis=1),
                        pd.get_dummies(df[col],
                        drop_first=True, prefix=col)], axis=1)
    return df


def create_family_feature(df):
    """
    :param df: input dataframe where the family feature needs
            to be added
    :return: None
    """
    def family(x):
        val = None
        if x == 1:
            val = "single"
        elif x == 2:
            val = "couple"
        elif x in [3,4,5]:
            val = "single parent"
        elif x in [6,7,8]:
            val = "family"
        elif x in [9,10,11]:
            val = "multiperson household"
        else:
            val = np.nan
        return val

    df['FAMILY'] = df['LP_FAMILIE_FEIN'].apply(lambda x: family(x))
    col = 'FAMILY'
    # df = pd.concat([df.drop([col], axis=1),
    #                 pd.get_dummies(df[col],
    #                 drop_first=True, prefix=col)], axis=1)
    le_family = LabelEncoder()
    fit_col = pd.Series([i for i in df.loc[:, col].unique() if type(i) == str])
    le_family.fit(fit_col)
    df[col] = df[col].map(lambda x: le_family.transform([x])[0] if type(x) == str else x)
    df.drop(['LP_FAMILIE_GROB'], axis=1, inplace=True)
    print('Created family feature')
    return df



def create_status_feature(df):
    """
    :param df: input dataframe where the status feature needs
            to be added
    :return: None
    """
    def family_status(x):
        val = None
        if x in [1,2]:
            val = "low-income earners"
        elif x in [3,4,5]:
            val = "average earners"
        elif x in [6,7]:
            val = "independants"
        elif x in [8,9]:
            val = "houseowners"
        elif x == 10:
            val = "top earners"
        else:
            val = np.nan
        return val

    df['FAMILY_STATUS'] = df['LP_STATUS_FEIN'].apply(lambda x: family_status(x))
    col = 'FAMILY_STATUS'
    # df = pd.concat([df.drop([col], axis=1),
    #                 pd.get_dummies(df[col],
    #                 drop_first=True, prefix=col)], axis=1)
    le_status = LabelEncoder()
    fit_col = pd.Series([i for i in df.loc[:, col].unique() if type(i) == str])
    le_status.fit(fit_col)
    df[col] = df[col].map(lambda x: le_status.transform([x])[0] if type(x) == str else x)
    df.drop(['LP_STATUS_GROB'], axis=1, inplace=True)
    print('Created status feature')
    return df


def create_features_from_cameo(df):
    """
    :param df: input dataframe where the features from CAMEO are to be added
    :return: None
    """
    col = 'CAMEO_INTL_2015'
    values = df[col]
    values = values.replace({'XX': 0, np.nan: 0})
    values = values.astype(np.int)
    wealth = values.apply(lambda x: math.floor(x / 10))
    wealth = wealth.replace({0: np.nan})
    lifestage = values.apply(lambda x: x % 10)
    lifestage = lifestage.replace({0: np.nan})
    df[col + '_WEALTH'] = wealth
    df[col + '_LIFESTAGE'] = lifestage
    df.drop(col, axis=1, inplace=True)
    print('Created features from CAMEO_INTL')


def unknown_to_nans(df):
    """
    :param df: dataframe where unknowns are to be replaced
    :return: df: dataframe where the unknowns are replaced wih NANs
    """
    dias_attributes = pd.read_excel('data/DIAS Attributes - Values 2017.xlsx', index_col=False, skiprows=1)
    dias_attributes.drop("Unnamed: 0", axis=1, inplace=True)
    dias_attributes.ffill(inplace=True)
    unknowns = dias_attributes.loc[dias_attributes.Meaning.str.contains('unknown')].copy()
    unknowns['Value'] = unknowns['Value'].map(str).str.split(',')
    for col in df.columns:
        if col in unknowns.Attribute.values:
            unk_vals = unknowns[unknowns.Attribute == col]['Value'].values[0]
            for val in unk_vals:
                df[col] = df[col].replace(int(val), np.nan)

    # exception processing
    if 'GEBURTSJAHR' in df.columns:
        df['GEBURTSJAHR'] = df['GEBURTSJAHR'].replace({0: np.nan})
    print("Unknowns processed to NANs")

    return df


def get_null_values_df(df):
    """
    :param df: dataframe where the null values need to be found
    :return: null_df: sorted dataframe with percent of nulls for each column
    """
    null_df = pd.DataFrame({'col_name':((df.isna().sum()/len(df))*100).index,
                        'null_percent':((df.isna().sum()/len(df))*100).values}) \
                        .sort_values(by=['null_percent'], ascending=False)
    return null_df


def get_skewed_columns(df):
    """
    :param df: dataframe where the skewed columns need to determined
    :return: skew_cols: dataframe with the skewed columns
    """
    skew_limit = 1  # define a limit above which we will log transform
    skew_vals = df.skew()
    # Showing the skewed columns
    skew_cols = (skew_vals
                 .sort_values(ascending=False)
                 .to_frame()
                 .rename(columns={0: 'Skew'})
                 .query('abs(Skew) > {}'.format(skew_limit)))
    return skew_cols


def detect_outliers(df, columns):
    """
    :param df: input data in the form of a dataframe
    :param columns: columns where the outliers need to be found
    :return: df_clean: cleaned dataframe where the outliers have been removed
    """

    # Select the indices for data points you wish to remove
    outliers_lst = []
    leave_cols = ['GEBURTSJAHR', 'MIN_GEBAEUDEJAHR']
    # For each feature find the data points with extreme high or low values
    for feature in columns:

        if feature not in leave_cols:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            step = 1.5 * (Q3 - Q1)

            # Display the outliers
            # print("Data points considered outliers for the feature '{}':".format(feature))

            # finding any points outside of Q1 - step and Q3 + step
            outliers_rows = df.loc[~((df[feature] >= Q1 - step) & (df[feature] <= Q3 + step)), :]
            # display(outliers_rows)

            outliers_lst.append(list(outliers_rows.index))

    outliers = list(itertools.chain.from_iterable(outliers_lst))
    # List of duplicate outliers
    dup_outliers = list(set([x for x in tqdm(outliers, position=0) if outliers.count(x) > 1]))
    df_clean = df.loc[~df.index.isin(dup_outliers)]
    print("Processed outliers")
    return df_clean


def impute_data(df, strategy="median"):
    """
    :param df: dataframe to be imputed
    :param strategy: strategy for imputation of values ('mean', 'median', 'most_frequent')
    :return: df_imputed: the imputed dataframe
    """

    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    df_imputed = pd.DataFrame(imputer.fit_transform(df))
    df_imputed.columns = df.columns
    print("Imputed dataframe")
    return df_imputed


# check columns with similar values
def get_similar_value_cols(df, percent=90):
    """
    :param df: input data in the form of a dataframe
    :param percent: integer value for the threshold for finding similar values in columns
    :return: sim_val_cols: list of columns where a singular value occurs more than the threshold
    """
    count = 0
    sim_val_cols = []
    for col in df.columns:
        percent_vals = (df[col].value_counts()/len(df)*100).values
        # filter columns where more than 90% values are same and leave out binary encoded columns
        if percent_vals[0] > percent and len(percent_vals) > 2:
            sim_val_cols.append(col)
            count += 1
    print("Total columns with majority singular value shares: ", count)
    return sim_val_cols


def create_features_from_lebensphase(df, column):
    life_stage = {1: 'younger_age', 2: 'middle_age', 3: 'younger_age',
                  4: 'middle_age', 5: 'advanced_age', 6: 'retirement_age',
                  7: 'advanced_age', 8: 'retirement_age', 9: 'middle_age',
                  10: 'middle_age', 11: 'advanced_age', 12: 'retirement_age',
                  13: 'advanced_age', 14: 'younger_age', 15: 'advanced_age',
                  16: 'advanced_age', 17: 'middle_age', 18: 'younger_age',
                  19: 'advanced_age', 20: 'advanced_age', 21: 'middle_age',
                  22: 'middle_age', 23: 'middle_age', 24: 'middle_age',
                  25: 'middle_age', 26: 'middle_age', 27: 'middle_age',
                  28: 'middle_age', 29: 'younger_age', 30: 'younger_age',
                  31: 'advanced_age', 32: 'advanced_age', 33: 'younger_age',
                  34: 'younger_age', 35: 'younger_age', 36: 'advanced_age',
                  37: 'advanced_age', 38: 'retirement_age', 39: 'middle_age',
                  40: 'retirement_age'}

    fine_scale = {1: 'low', 2: 'low', 3: 'average', 4: 'average', 5: 'low', 6: 'low',
                  7: 'average', 8: 'average', 9: 'average', 10: 'wealthy', 11: 'average',
                  12: 'average', 13: 'top', 14: 'average', 15: 'low', 16: 'average',
                  17: 'average', 18: 'wealthy', 19: 'wealthy', 20: 'top', 21: 'low',
                  22: 'average', 23: 'wealthy', 24: 'low', 25: 'average', 26: 'average',
                  27: 'average', 28: 'top', 29: 'low', 30: 'average', 31: 'low',
                  32: 'average', 33: 'average', 34: 'average', 35: 'top', 36: 'average',
                  37: 'average', 38: 'average', 39: 'top', 40: 'top'}

    df['LP_LEBENSPHASE_FEIN_AGE'] = df[column].map(life_stage)
    df['LP_LEBENSPHASE_FEIN_INCOME'] = df[column].map(fine_scale)
    cols = ['LP_LEBENSPHASE_FEIN_AGE', 'LP_LEBENSPHASE_FEIN_INCOME']
    for col in cols:
        le_status = LabelEncoder()
        fit_col = pd.Series([i for i in df.loc[:, col].unique() if type(i) == str])
        le_status.fit(fit_col)
        df[col] = df[col].map(lambda x: le_status.transform([x])[0] if type(x) == str else x)
    # df.drop(column, axis=1, inplace=True)
    print('Created lebensphase features')


def create_youth_movement_features(df):
    """
    :param df: input dataframe from where the features need to be created
    :return: None
    """
    # decade mapper
    decade_map = {}
    decade_keys = np.arange(1, 16)
    decade_vals = [40, 40, 50, 50, 60, 60, 60, 70, 70, 80, 80, 80, 80, 90, 90]
    decade_map.update(zip(decade_keys, decade_vals))
    df['PRAEGENDE_JUGENDJAHRE_DECADE'] = df['PRAEGENDE_JUGENDJAHRE'].map(decade_map)

    # movement: mainstream or avantgarde
    movement_map = {}
    movement_keys = np.arange(1, 16)
    # 1 for mainstream and 0 for avantgarde
    movemment_vals = [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    movement_map.update(zip(movement_keys, movemment_vals))
    df['PRAEGENDE_JUGENDJAHRE_MOVEMENT'] = df['PRAEGENDE_JUGENDJAHRE'].map(movement_map)
    # drop the original column
    df.drop(['PRAEGENDE_JUGENDJAHRE'], axis=1, inplace=True)
    print("Created youth movement features")


def handle_mixed_dtypes(value):
    """
    :param value: string or int from dataframe column
    :return: converted integer value
    """
    try:
        value = int(value)
    except:
        value = np.nan
    return value


def encode_categorical_columns(df, columns, unk_tokens):
    """
    :param df: input data in the form of a dataframe
    :param columns (list of strings): column to be encoded
    :param unk_tokens (list of strings): token to be classified as NAN
    :return: None
    """
    for i in range(len(columns)):
        if unk_tokens[i]:
            # encode unknown category as NAN
            df[columns[i]] = df[columns[i]].apply(lambda x: np.nan if x == unk_tokens[i] else x)
        # encode categories
        le = LabelEncoder()
        fit_col = pd.Series([i for i in df.loc[:, columns[i]].unique() if type(i) == str])
        le.fit(fit_col)
        df[columns[i]] = df[columns[i]].map(lambda x: le.transform([x])[0] if type(x) == str else x)
        print("Encoded column: {}".format(columns[i]))


def handle_datetime_column(df, column):
    """
    :param df: input dataframe
    :param column: column to be converted
    :return: None
    """
    # convert datetime to get day, month and year of reception
    df[column+'_DAY'] = pd.to_datetime(df[column]).dt.day
    df[column+'_MONTH'] = pd.to_datetime(df[column]).dt.month
    df[column+'_YEAR'] = pd.to_datetime(df[column]).dt.year
    df.drop([column], axis=1, inplace=True)
    print("Converted datetime column to day, month and year columns")


def drop_cols_with_nulls(df, null_df, null_threshold=30):
    """
    :param df: dataframe where columns need to be dropped
    :param null_df: dataframe with null percentages of columns
    :param null_threshold: threshold for dropping nulls
    :return: None
    """
    # get cols with more than 30% nulls
    drop_cols = null_df[null_df.null_percent > null_threshold]['col_name'].values
    df.drop(drop_cols, axis=1, inplace=True)
    print('Dropped columns with most nulls')


# feature binning function
def bin_column(df, column, bins, labels, include_lowest=True, right=True):
    """
    Takes in a column name, bin cut points and labels, replaces the original column with a
    binned version, and replaces nulls (with 'unknown' if unspecified).
    :param df: input dataframe
    :param column: column to be binned
    :param bins: list of bins
    :param labels: list of labels for the bins
    :param include_lowest: bool value indicates whether bins includes the leftmost edge or not
    :param right: bool value indicates whether bins includes the rightmost edge or not
    :return: None
    """
    values = pd.cut(df[column], bins=bins, labels=labels, include_lowest=include_lowest, right=right).values
    df[column] = values
    print("Binned ", column)


# test your functions here
if __name__ == "__main__":
    df = pd.read_csv('data/mailout_train.csv', sep=';')
    bin_column(df, 'ANZ_PERSONEN', [0, 1, 2, 3, 4, 5, df['ANZ_PERSONEN'].max() + 1], False,
               include_lowest=True, right=False)
    dias_attributes = pd.read_excel('data/DIAS Attributes - Values 2017.xlsx', index_col=False, skiprows=1)
    dias_attributes.drop("Unnamed: 0", axis=1, inplace=True)
    dias_attributes.ffill(inplace=True)
    num_cols = dias_attributes[dias_attributes.Meaning.str.contains('numeric')]['Attribute'].values
    detect_outliers(df, num_cols)


