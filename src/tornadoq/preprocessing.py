from tornadoq.shadows import generate_shadows, extend_features
from torch.utils.data import Dataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import torch
from pathlib import Path
import time


from pathlib import Path
import pandas as pd

def _read_table(path: str, percent_data: float = 1.0) -> pd.DataFrame:
    if not (0.0 <= percent_data <= 1.0):
        raise ValueError("percent_data must be between 0 and 1")

    ext = Path(path).suffix.lower()

    if ext == ".csv":
        if percent_data == 1.0:
            return pd.read_csv(path)

        # Count total rows (minus header)
        with open(path, "r") as f:
            total_rows = sum(1 for _ in f) - 1

        rows_to_read = int(total_rows * percent_data)

        return pd.read_csv(path, nrows=rows_to_read)

    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path)

        if percent_data == 1.0:
            return df

        rows_to_keep = int(len(df) * percent_data)
        return df.iloc[:rows_to_keep]

    else:
        raise ValueError(f"Unsupported file type: {ext}")

def _read_table_debug(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()

    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path, nrows=10)
    elif ext == ".csv":
        df = pd.read_csv(path, nrows=10)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    print(f"\nDEBUG READ: {path}")
    print("Columns:", df.columns.tolist())
    print("First row values:")
    print("Dict", df.iloc[0].to_dict())

    return df

def DataMaker(TRAIN_FILE, 
              TEST_FILE, 
              VALIDATION_FILE, 
              withShadows=False, 
              output_filename=None, 
              shadow_options = None, 
              debug = False, 
              percent_data = 1.0):
    
    if debug == True:
        df_train = _read_table_debug(TRAIN_FILE) 
        df_test  = _read_table_debug(TEST_FILE)
        df_val   = _read_table_debug(VALIDATION_FILE)
            
    else:
        df_train = _read_table(TRAIN_FILE, percent_data) 
        df_test  = _read_table(TEST_FILE, percent_data)
        df_val   = _read_table(VALIDATION_FILE, percent_data)

    if withShadows:
        def apply_shadows(df):
            shadow_df = generate_shadows(df, **shadow_options,)
            return extend_features(shadow_df, df)
        start = time.time()
        df_train = apply_shadows(df_train)
        df_test  = apply_shadows(df_test)
        df_val   = apply_shadows(df_val)
        end = time.time()
        print(f"Duration of Shadow Generation: {(end-start):.2f}.")
        
        number_of_datapoints = len(df_test) + len(df_train)+len(df_val)
        print(f"Duration of Shadow Generation per circuit: {((end-start)/number_of_datapoints):.2f}.")

        
    if output_filename:
        combined_df = pd.concat([df_train, df_val, df_test], ignore_index=True)
        combined_df.to_csv(output_filename, index=False)
        print(f"✓ Combined data saved: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")

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

class LUQPIDataset(Dataset):
    """Yields (x_orig, x_shadow, y) triples for LUQPI training."""
    def __init__(self, X_orig, X_shadow, y):
        def _to_tensor(a):
            if hasattr(a, 'values'):
                a = a.values
            return torch.from_numpy(a.copy()).float()
        self.X_orig   = _to_tensor(X_orig)
        self.X_shadow = _to_tensor(X_shadow)
        self.y        = _to_tensor(y if not hasattr(y, 'values') else y.values)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_orig[idx], self.X_shadow[idx], self.y[idx]


def Preprocess(df_train, df_test, df_val, balance=None, classes='binary'):
    # Separate features and targets
    X_train = df_train.drop(['ef_class', 'ef_binary'], axis=1, errors='ignore')
    X_test  = df_test.drop(['ef_class', 'ef_binary'], axis=1, errors='ignore')
    X_val   = df_val.drop(['ef_class', 'ef_binary'], axis=1, errors='ignore')

    # ✅ Fix: ensure all feature names are strings (important for sklearn)
    X_train.columns = X_train.columns.astype(str)
    X_test.columns  = X_test.columns.astype(str)
    X_val.columns   = X_val.columns.astype(str)

    # Targets
    y_train_binary = df_train['ef_binary']
    y_test_binary  = df_test['ef_binary']
    y_val_binary   = df_val['ef_binary']

    y_train_class  = df_train['ef_class']
    y_test_class   = df_test['ef_class']
    y_val_class    = df_val['ef_class']

    if classes == 'binary':
        y_train, y_test, y_val = y_train_binary, y_test_binary, y_val_binary
    elif classes == 'multiclass':
        y_train, y_test, y_val = y_train_class, y_test_class, y_val_class
    else:
        raise ValueError(f"classes parameter must be either binary or multiclass. You have {classes}.")

    # Impute
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imputed  = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    X_val_imputed   = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)

    # Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled  = scaler.transform(X_test_imputed)
    X_val_scaled   = scaler.transform(X_val_imputed)

    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test  = pd.DataFrame(X_test_scaled,  columns=X_test.columns)
    X_val   = pd.DataFrame(X_val_scaled,   columns=X_val.columns)

    # SMOTE (train only)
    if balance == 'smote':
        smote_class = SMOTE(random_state=42, k_neighbors=3)
        X_train, y_train = smote_class.fit_resample(X_train, y_train)

    return X_train, y_train, X_test, y_test, X_val, y_val


