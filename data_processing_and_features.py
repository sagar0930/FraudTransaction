import pandas as pd

def convert_dtpye(df):
    num_cols = df [['step', 'amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFlaggedFraud', 'isFraud']]
    for col in num_cols:
         df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def handling_blanks(df):
    df['type'] = df['type'].replace('', 'NOTYPE')
    df['nameOrig'] = df['nameOrig'].replace('', 'NONAME')
    return df

def handing_null(df):
    df[~((df['oldbalanceDest'].isnull() & (df['isFraud'] == 0)) | (df['newbalanceOrig'].isnull() & (df['isFraud'] == 0)))]
    condition = df['type'].isin(['DEBIT', 'CASH_OUT', 'TRANSFER', 'PAYMENT', 'NOTYPE'])
    df.loc[condition, 'newbalanceOrig'] = df.loc[condition, 'oldbalanceOrg'] - df.loc[condition, 'amount']
    mask = (df['type'] == 'CASH_IN')
    df.loc[mask, 'newbalanceOrig'] = df.loc[mask, 'oldbalanceOrg'] + df.loc[mask, 'amount']
    df['oldbalanceDest'] = df['oldbalanceDest'].fillna(0.00)
    return df

def cat_feature(df):
    df_dum = pd.get_dummies(df['type'], drop_first=True).astype(int)
    df = pd.concat([df, df_dum], axis = 1)
    df.drop(['type', 'nameDest', 'nameOrig', 'step'], axis=1, inplace=True)
    return df