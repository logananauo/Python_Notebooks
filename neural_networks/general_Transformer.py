
import torch
import torch.nn as nn


class Transformer(nn.Module):

    def __init__(self, input_size, d_model, nhead, num_layers, num_classes):
        super().__init__()

        # convert input features into the model's internal dimension
        self.input_projection = nn.Linear(
            input_size,
            d_model
        )

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # final classification layer
        self.fc = nn.Linear(
            d_model,
            num_classes
        )

    def forward(self, x):

        # project input into Transformer dimension
        x = self.input_projection(x)

        # process sequence with Transformer
        x = self.transformer(x)

        # use final time step
        x = x[:, -1, :]

        # classification
        x = self.fc(x)

        return x


### example configuration
model = Transformer(
    input_size=10,
    d_model=64,
    nhead=4,
    num_layers=2,
    num_classes=5
)

print(model)
