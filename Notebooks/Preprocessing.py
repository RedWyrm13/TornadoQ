import pandas as pd
import torch

# Dataset reading
def DataMaker(TRAIN_FILE, TEST_FILE, VALIDATION_FILE, flag = None)
    
    # Load training data
    df_train = pd.read_excel(TRAIN_FILE)
    # Load test data
    df_test = pd.read_excel(TEST_FILE)
    # Load validation data
    df_val = pd.read_excel(VALIDATION_FILE)

    # Flag specifies what random shadows circuit to use (add flags/ circuits as desired)
    if flag == "RS_full_ent":

        #### This should be replaced by actually generating the random shadows feature engineering ###############
        # # Quantum augmented datasets using random shadows
        # EXTRA_TRAIN = "../Data/32_featuresXY_train_full_ent.csv"
        # EXTRA_TEST  = "../Data/32_featuresXY_test_fullent.csv"
        # EXTRA_VALID = "../Data/32_featuresXY_valid_fullent.csv"
        
        # # Extra features
        # extra_train_df = pd.read_csv(EXTRA_TRAIN)
        # extra_test_df = pd.read_csv(EXTRA_TEST)
        # extra_valid_df = pd.read_csv(EXTRA_VALID)
        ###################################################################################
        # Constructs extra_train_df, extra_test_df, extra_valid_df
        
        # Drop first column by index (Assuming the numbering is still in place)
        extra_train_df = extra_train_df.drop(extra_train_df.columns[0], axis=1)
        extra_test_df = extra_test_df.drop(extra_test_df.columns[0], axis=1)
        extra_valid_df = extra_valid_df.drop(extra_valid_df.columns[0], axis=1)
        
        # Concatenate extra features (axis=1 for columns)
        df_train = pd.concat([df_train, extra_train_df], axis=1)
        df_test  = pd.concat([df_test, extra_test_df], axis=1)
        df_valid  = pd.concat([df_valid, extra_valid_df], axis=1)
    
    print(f"✓ Training data loaded: {df_train.shape[0]} rows, {df_train.shape[1]} columns")
    print(f"✓ Test data loaded: {df_test.shape[0]} rows, {df_test.shape[1]} columns")

    return df_train, df_test, df_val

# Dataset Preprocessing
class ClassificationDataset(Dataset):

    def __init__(self, X, y):

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.values
            
        self.X = torch.from_numpy(X.copy()).float()
        self.y = torch.from_numpy(y.copy()).float()
        
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
   
def Preprocess(df_train, df_test, df_val, balance = None, classes = 'binary'):

    # Separate features and targets
    X_train = df_train.drop(['ef_class', 'ef_binary'], axis=1, errors='ignore')
    X_test = df_test.drop(['ef_class', 'ef_binary'], axis=1, errors='ignore')
    X_val = df_val.drop(['ef_class', 'ef_binary'], axis=1, errors='ignore')
    
    y_train_binary = df_train['ef_binary']
    y_test_binary = df_test['ef_binary']
    y_val_binary = df_val['ef_binary']
    
    y_train_class = df_train['ef_class']
    y_test_class = df_test['ef_class']
    y_val_class = df_val['ef_class']

    if classes == 'binary':
        y_train = y_train_binary
        y_test = y_test_binary
        y_val = y_val_binary
    else:
        y_train = y_train_class
        y_test = y_test_class
        y_val = y_val_class

    # Handle missing values - TO CHECK LATER (drop or get median for the subclass)
    imputer = SimpleImputer(strategy='mean')

    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    X_val_imputed = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)

    # Normalize features
    # use for -1 to 1
    #scaler = StandardScaler()

    #use for 0 to 1
    scaler= MinMaxScaler(feature_range=(0, 1))
    
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    X_val = pd.DataFrame(X_val_scaled, columns=X_val.columns)

    if balance == 'smote':
        smote_class = SMOTE(random_state=42, k_neighbors=3)
        X_train, y_train = smote_class.fit_resample(X_train, y_train)

    return X_train, y_train, X_test, y_test, X_val, y_val

