import torch.nn as nn


class SimpleFcModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, device="cuda", dropout=0, use_batch_norm=False):
        super(SimpleFcModel, self).__init__()

        self.device = device
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, int(output_size))

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.bn1 = nn.LayerNorm(hidden_size) if use_batch_norm else nn.Identity()
        self.bn2 = nn.LayerNorm(hidden_size) if use_batch_norm else nn.Identity()
        self.bn3 = nn.LayerNorm(hidden_size) if use_batch_norm else nn.Identity()
        # self.bn3 = nn.BatchNorm1d(hidden_size) if use_batch_norm else nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        x = self.dropout2(x)
        #x = self.bn3(self.fc3(x))
        #x = self.relu(x)
        #x = self.dropout3(x)
        x = self.fc4(x)

        return x


