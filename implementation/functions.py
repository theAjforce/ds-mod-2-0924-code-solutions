import pandas as pd
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
import pickle
import sklearn
from sklearn.impute import SimpleImputer

#read all csv files used
def one_fell_reader(file_names): #as list of 'string.csv'
    if type(file_names) == list:
        df_list = []
        for file in file_names:
            df = pd.read_csv(file)
            df_list.append(df)
        return df_list
    else:
        df = pd.read_csv(file_names)
        return df

#dropping columns that have 50% or more null values
def one_fell_dropper(df):
    threshold = len(df) * 0.6 #creating the drop threshold
    df = df.dropna(thresh=threshold,axis=1)
    return df

#filling nans with OneFellImputer
def one_fell_imputer(df,test=False):
    if test == False:
        col_dict = {}

        for col in df.select_dtypes(include=['object']).columns:
            cat_imp = SimpleImputer(strategy='most_frequent')
            cat_imp.fit(df[[col]])
            col_dict[col] = cat_imp
            df[col]= cat_imp.transform(df[[col]]).reshape(1,-1)[0]
        for col in df.select_dtypes(include=['int64','float64']).columns:
            num_imp = SimpleImputer(strategy='mean')
            num_imp.fit(df[[col]])
            col_dict[col] = num_imp
            df[col] = num_imp.transform(df[[col]])
        with open('imputer_types.pickle','wb') as f:
            pickle.dump(col_dict,f)
    else:
        with open('imputer_types.pickle','rb') as f:
                imps = pickle.load(f)
        for col in df:
            df[col] = imps[col].transform(df[[col]])
    return df


#encoding all object type columns with OneFellEncoder 
def one_fell_encoder(df,target,tar_type,file_name,test=False):
#df = entire data frame, range = length of target column, tar_type = TargetEncoder target_type, target = target column for TargetEncode
    col_object = df.select_dtypes(include=['object']).columns
    if test == False:
        col_dict = {}  
        for col in col_object:
            if  df[col].nunique() == 2:  #binary encoding columns with 2 categories
                col_dict[col] = 'binary'
                cat_names = df[col].unique()
                df[col] = df[col].apply(lambda x: 1 if x==cat_names[0] else 0)
            elif df[col].nunique() < 10:  #OneHotEncoding columns with less than 10 categories
                cat_names = list(df[col].unique())
                ohe = OneHotEncoder(categories=[cat_names], sparse_output=False)
                ohe.fit(df[[col]])
                col_names = ohe.get_feature_names_out()
                vals = pd.DataFrame(ohe.transform(df[[col]]), columns=col_names)
                col_dict[col] = ohe
                df = pd.concat([df.reset_index(drop=True), vals], axis=1)
                df.drop(col, axis=1, inplace=True)
            else:  #TargetEncoding all other columns
                tar = TargetEncoder(target_type=tar_type)
                y_train = df[[target]]
                x_train = df[[col]]
                x = df[[col]]
                tar.fit(x_train, y_train)
                col_dict[col] = tar
                df[col] = tar.transform(x)
        with open(file_name,'wb') as f:
            pickle.dump(col_dict,f)
    else:
        with open(file_name,'rb') as f:
                enc = pickle.load(f)
        for col in col_object:
            if type(enc[col]) == sklearn.preprocessing.TargetEncoder:
                df[col] = enc[col].transform(df[[col]]) 
            elif type(enc[col]) == sklearn.preprocessing._encoders.OneHotEncoder:
                col_names = enc[col].get_feature_names_out()
                vals = pd.DataFrame(enc[col].transform(df[[col]]), columns=col_names)
                df = pd.concat([df.reset_index(drop=True), vals], axis=1)
                df.drop(col, axis=1, inplace=True)
            else:
                cat_names = df[col].unique()
                df[col] = df[col].apply(lambda x: 1 if x==cat_names[0] else 0)  
    return df

#train test split df for validation
def one_fell_splitter(x,y,test_size,random_state):
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size,random_state=random_state)
    return x_train, x_test, y_train, y_test


#building xgb model
def one_fell_modeler(x_train=None, y_train=None, x_test=None, y_test=None, test=None):
    if (test is None) and (x_train is not None) and (y_train is not None):
        if (y_train.dtype != 'Object') and (y_train.nunique() > 20):
            xgb = XGBRegressor()
            xgb.fit(x_train,y_train)
            with open ('xgb_model.pickle','wb') as f:
                pickle.dump(xgb,f)
            preds = xgb.predict(x_test)
            rmse = mean_squared_error(y_test,xgb.predict(x_test),squared=False)
            print(f"Root Mean Squared Error is {rmse}")
        else:
            xgb = XGBClassifier()
            xgb.fit(x_train,y_train)
            with open ('xgb_model.pickle','wb') as f:
                pickle.dump(xgb,f)                     
    elif test is not None:
        with open ('xgb_model.pickle','rb') as f:
            xgb = pickle.load(f)
        preds = xgb.predict(test)
        test['target'] = preds
        test = test['target']
        return test
    else:
        raise ValueError("If test is False, x_train and y_train must have values.")
    return xgb