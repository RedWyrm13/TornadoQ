from typing import List
from tornadoq.preprocess import load_csv
from tornadoq.shadows import generate_shadows, extend_features

def load_all_data(filenames: List[str], withShadows: bool = False, filesave_name: str = None):
    """
    filenames: a list of file paths. Each element should be a a full path to the csv
    withShadows: Boolean variable. If true, the returned dataframes include the shadows, else they are just classical features
    filesave_name: This saves the shadows only csv to the specified location
    """

    # Contains dataset name as key (test, train, or val), and the actual pandas df as the value
    dfs = {}
    for filename in filenames:
        df = load_csv(filename)

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

    return dfs
            