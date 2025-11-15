# All imports
from Imports import *
from Preprocessing import *
from Helper import *
import train
import models
import loadSaveEval

# Load dataset and device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data paths
TRAIN_FILE = '../Data/2025-Quantathon-Tornado-Q-training_data-640-examples.xlsx'
TEST_FILE = '../Data/2025-Quantum-Tornado-Q-test_data-200-examples.xlsx'
VALIDATION_FILE = '../Data/2025-Quantum-Tornado-validation_data-160-examples.xlsx'

# Create pandas dataframes
df_train, df_test, df_val = DataMaker(TRAIN_FILE, TEST_FILE, VALIDATION_FILE, flag = "RS_full_ent")

# Processing
X_train, y_train, X_test, y_test, X_val, y_val = Preprocess(df_train, df_test, df_val, balance = 'smote', classes = 'binary')

train_data = ClassificationDataset(X_train, y_train)
validation_data = ClassificationDataset(X_val, y_val)
test_data = ClassificationDataset(X_test, y_test)

# Dataloaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)
val_loader = DataLoader(validation_data, batch_size=64, shuffle=False, drop_last=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, drop_last=True)

print("Number of training samples:", len(train_data))
print("Number of validation samples:", len(validation_data))
print("Number of test samples:", len(test_data))

# model options here...
model = "DNN"
n_epochs = 10
lr = .01

# Initialize, train, and evaluate model
model = InitializeModel(model, load_path = None, classifier = "Binary", input_size = df_train.shape[1])
train(model, n_epochs, lr, train_loader, val_loader)
eval_model(model, test_loader)
save(model)
# load(model, savepoint)