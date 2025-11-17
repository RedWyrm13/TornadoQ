from typing import List
from tornadoq.shadows import generate_shadows, extend_features
import os
import pandas as pd

def load_all_data(filenames: List[str], withShadows: bool = False, filesave_name: str = None):
    """
    filenames: a list of file paths. Each element should be a a full path to the csv
    withShadows: Boolean variable. If true, the returned dataframes include the shadows, else they are just classical features
    filesave_name: This saves the shadows only csv to the specified location
    """

    # Contains dataset name as key (test, train, or val), and the actual pandas df as the value
    dfs = {}
    for filename in filenames:
        
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".csv":
            df = pd.read_csv(filename)
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(filename)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        # Feature engineering with random shadows if this flag is true
        if withShadows:
            shadow_df = generate_shadows(df, filename_save = filesave_name)
            df = extend_features(shadow_df, df)

        if 'train' in filename.lower():
            dfs['train'] = df
        elif 'val' in filename.lower():
            dfs['val'] = df
        elif 'test' in filename.lower():
            dfs['test'] = df
        else:
            raise ValueError("filename must contain identifiable name. Either 'train', 'val' or 'test' ")
            
    print(f"✓ Training data loaded: {dfs['train'].shape[0]} rows, {dfs['train'].shape[1]} columns")
    print(f"✓ Test data loaded: {dfs['test'].shape[0]} rows, {dfs['test'].shape[1]} columns")
    print(f"✓ Validation data loaded: {dfs['val'].shape[0]} rows, {dfs['val'].shape[1]} columns")
    
    return dfs
            