def InitializeModel(model, load_path = None, classifier = "Binary", input_size = 8):

    ############ Conditionals for all model types ############## 
    ### DNN Various Input ###
    if model == "DNN":
        #Initialize architecture here
        if classifier = "Binary":
            model = BinaryDNN(input_size)
        if classifier = "Multiclass":
            model = MulticlassDNN(input_size)

    ### PQC Models ###
    else if model == "RandomLayer":
        #Initialize PQC
        from QuantumCircuits import InitializePQC
        InitializePQC(model)

        #Initialize architecture here
        if classifier = "Binary":
            model = 
        if classifier = "Multiclass":
            model = 

    else if model == "StronglyEntangled":
        #Initialize PQC
        from QuantumCircuits import InitializePQC
        InitializePQC(model)

        #Initialize architecture here
        if classifier = "Binary":
            model = BinaryPQC()
        if classifier = "Multiclass":
            model = MulticlassPQC()

    else:
        print("Not a valid model choice...")
        return

    if load_path != None:
        from load import load
        load(model, load_path)
        
    return model

# ====              Model Architectures                  ====
# ===========================================================
#
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
    def __init__(self):
        super().__init__()

        # Encodes features from dataset
        self.feature_encoder = QuantumFeatureEmbeddingBatch()

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
    def __init__(self):
        super().__init__()

        # Encodes features from dataset
        self.feature_encoder = QuantumFeatureEmbeddingBatch()

        # Classifies based on encoded features
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )

    def forward(self, features):
        feats_encoded = self.feature_encoder(features)
        class_probs = self.classifier(feats_encoded.float())

        return class_probs  # Shape: (batch_size, 1)

#############################################################
