
import torch
import torch.nn as nn



class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()

        # RNN layer
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # fully connected output layer
        self.fc = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, x):

        # RNN processes the sequence
        output, hidden = self.rnn(x)

        # Use the final time step
        x = output[:, -1, :]

        # classification
        x = self.fc(x)

        return x


### example configuration
model = RNN(
    input_size=10,
    hidden_size=64,
    num_layers=2,
    num_classes=5
)

print(model)
