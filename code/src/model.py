import torch
import torch.nn as nn

#### This is the basic feedforward model ####
# Create MLP model in a class with customable number of layers and hidden units
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units, num_layers):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        # Create input layer
        self.layers.append(nn.Linear(self.input_dim, self.hidden_units))
        
        # Create hidden layers
        for i in range(self.num_layers):
            self.layers.append(nn.Linear(self.hidden_units, self.hidden_units))
        
        # Create output layer
        self.layers.append(nn.Linear(self.hidden_units, self.output_dim))
        
    def forward(self, x):
        #flatten the input
        x = x.view(-1, self.input_dim)
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x) #without ReLu activation function
        return x
    
# Encoder model

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        pass

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        pass

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.layer = nn.ModuleList()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class VAEskip(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAEskip, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.layer = nn.ModuleList()

    def forward(self, x):
        x0 = x
        x = self.encoder(x)
        x = self.decoder(x)
        x = x + x0
        return x
