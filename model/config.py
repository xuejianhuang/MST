import torch
class base_Config():
    def __init__(self):
        # Basic configurations
        self.model_name = "MST"  # Model name
        self.embedding_dim = 128  # Dimension of word embeddings
        self.hidden_dim = 128  # Dimension of hidden layers
        self.n_class = 2  # Number of classes
        self.n_hidden = 3  # Number of hidden layers
        self.dropout = 0.2  # Dropout probability
        self.weight_decay = 1e-4  # Weight decay for regularization
        self.patience = 5  # Patience for early stopping
        self.train = 0.7  # Train data split ratio
        self.val = 0.1  # Validation data split ratio
        self.test = 0.2  # Test data split ratio
        self.model_saved_path = '../model_saved/'  # Path to save trained models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device for training

        self.text_max_length = 30  # Maximum length of input text (e.g., maximum number of tokens)
        self.bert_freeze = False  # Freeze BERT layers during training
        self.embedding_freeze = False  # Freeze word embeddings during training
        self.fc_hidden_dim = 512  # Dimension of hidden layers in fully connected layers
        self.tf = 'concat'  # Type of fusion method: 'attention' or 'concat'
