from tornadoq.pqc import InitializePQC
import torch.nn as nn

def InitializeModel(model, load_path = None, classifier = "binary", input_size = 8, device = None):

    ############ Conditionals for all model types ############## 
    ### DNN Various Input ###
    if model == "DNN":
        #Initialize architecture here
        if classifier == "binary":
            model = BinaryDNN(input_size)
        if classifier == "multiclass":
            model = MulticlassDNN(input_size)

    ### PQC Models ###
    elif model in ("RandomLayer", "StronglyEntangling"):
        
        #Initialize PQC
        pqc = InitializePQC(model, device = device)

        #Initialize architecture here
        if classifier == "binary":
            model = BinaryPQC(pqc)
        if classifier == "multiclass":
            model = MulticlassPQC(pqc) 

    else:
        print("Not a valid model choice...")
        return

    if load_path != None:
        from tornadoq.loadSaveEval import load
        load(model, load_path)
        
    return model

# ====              Model Architectures                  ====
# ===========================================================
#
#############################################################
# LUQPI DNN
# Main branch uses only original features — the shadow head is an
# auxiliary reconstruction target that forces the encoder to learn
# quantum-relevant structure during training, then is discarded at inference.
class BinaryDNN_LUQPI(nn.Module):
    def __init__(self, n_orig, n_shadow):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_orig, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.shadow_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, n_shadow),
        )

    def forward(self, x_orig, x_shadow=None):
        h = self.encoder(x_orig)
        pred = self.classifier(h)
        if x_shadow is not None:
            return pred, self.shadow_head(h)
        return pred


#############################################################
# Binary DNN - Variable Input
class BinaryDNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        # Encodes features from dataset
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )
        # Classifies based on encoded features
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, features):
        feats_encoded = self.feature_encoder(features)
        class_probs = self.classifier(feats_encoded)

        return class_probs  # Shape: (batch_size, 1)

#############################################################
# Multiclass DNN - Variable Input
class MulticlassDNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        # Encodes features from dataset
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
        )
        # Classifies based on encoded features
        self.classifier = nn.Sequential(
            nn.Linear(1024, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 4)
        )
    def forward(self, features):
        feats_encoded = self.feature_encoder(features)
        class_target = self.classifier(feats_encoded)
        
        return class_target  # Shape: (batch_size, 4)

#############################################################
# Binary PQC - Variable Circuit
class BinaryPQC(nn.Module):
    def __init__(self, QuantumFeatureEmbeddingBatch):
        super().__init__()

        # Encodes features from dataset
        self.feature_encoder = QuantumFeatureEmbeddingBatch

        # Classifies based on encoded features
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        feats_encoded = self.feature_encoder(features)
        class_probs = self.classifier(feats_encoded.float())

        return class_probs  # Shape: (batch_size, 1)

#############################################################
# Multiclass PQC - Variable Circuit
class MulticlassPQC(nn.Module):
    def __init__(self, QuantumFeatureEmbeddingBatch):
        super().__init__()

        # Encodes features from dataset
        self.feature_encoder = QuantumFeatureEmbeddingBatch

        # Classifies based on encoded features
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 4)
        )

    def forward(self, features):
        feats_encoded = self.feature_encoder(features)
        class_probs = self.classifier(feats_encoded.float())
        
        return class_probs  # Shape: (batch_size, 4)

#############################################################
