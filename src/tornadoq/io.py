from __future__ import annotations
import logging
import pandas as pd


logger = logging.getLogger(__name__)




def load_excel_pair(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and test data from Excel files."""
    df_train = pd.read_excel(train_path)
    df_test = pd.read_excel(test_path)
    logger.info("Loaded train: %s rows, %s cols", df_train.shape[0], df_train.shape[1])
    logger.info("Loaded test : %s rows, %s cols", df_test.shape[0], df_test.shape[1])
    return df_train, df_test