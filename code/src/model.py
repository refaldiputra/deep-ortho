import torch
import torch.nn as nn


class MLP(nn.Module):
    '''
    MLP stands for Multi Layer Perceptron
    It is a feedforward neural network with at least 3 layers: input, hidden, and output
    It can serve as autoencoder as well, we assume the latent before the last layer
    '''
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

class ConvNet(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding, pool_size):
        super(ConvNet, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )
class NNNorm(nn.Sequential):
    def __init__(self, in_features, out_features):
        super(NNNorm, self).__init__(
            nn.Linear(in_features, out_features),
#            nn.BatchNorm1d(out_features),
            nn.ReLU()
        )
class DeconvNet(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding, scale):
        super(DeconvNet, self).__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding = padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=scale, mode='nearest')
        )

class DeconvNet2(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DeconvNet2, self).__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding = padding),
            nn.Sigmoid(),
        )

# Encoder block
class Encoder(nn.Sequential):
    def __init__(self, config):
        super(Encoder, self).__init__(
        ConvNet(**config['block1']),
        ConvNet(**config['block2']),
        ConvNet(**config['block3'])
        )
# Decoder block
class Decoder(nn.Sequential):
    def __init__(self, config):
        super(Decoder, self).__init__(
        DeconvNet(**config['block1']),
        DeconvNet(**config['block2']),
        DeconvNet(**config['block3']),
        DeconvNet2(**config['block4'])
        )

class VAE(nn.Module):
    '''
    VAE stands for Variational Autoencoder which consists of encoder and decoder
    Its task is to reconstruct the input data by training
    A signature of the autoencoder is to have a bottleneck in the middle
    We assume the latent representation z to be in the middle (the output of the encoder)
    '''
    def __init__(self,configs):
        super(VAE, self).__init__()
        self.encoder = Encoder(configs['encoder'])
        self.mid =NNNorm(**configs['mid'])
        self.decoder = Decoder(configs['decoder'])

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(z.size(0), -1)
        z = self.mid(z)
        z = z.view(z.size(0), -1, 4, 4) # 4 is fixed here to create a compressed representation
        x_ = self.decoder(z)
        return x_, z
    
class VAEskip(nn.Module):
    '''
    VAE with skip connection
    '''
    def __init__(self, configs):
        super(VAEskip, self).__init__()
        self.encoder = Encoder(configs['encoder'])
        self.mid =NNNorm(**configs['mid'])
        self.decoder = Decoder(configs['decoder'])

    def forward(self, x):
        x0 = x
        z = self.encoder(x)
        z = z.view(z.size(0), -1)
        z = self.mid(z)
        z = z.view(z.size(0), -1, 4, 4)
        x_ = self.decoder(z)
        x_ = x_ + x0 #skip connection
        return x_, z

#### Below is the important part from the paper ####

class OrthogonalProjector(nn.Module):
    # This is put in the middle of the encoder and decoder after 'mid'
    # We assume the input and output dimension as the same
    # Denote W as the weight matrix of the linear layer
    # We calculate the svd of Z = UDV^T
    # We select the first k' of V, I assume k'=k and no zero singular value
    # Assuming the batchsize is larger than the dimension of the representation
    # Then arrange it accordingly, to get the weight new weight W' = V^T D^-1 (column average)
    # Then, the output will be W'Z where Z is the batch input
    def __init__(self, dim):
        super(OrthogonalProjector, self).__init__()
        self.input_dim = dim
        self.output_dim = dim
        self.linear = nn.Linear(self.input_dim, self.output_dim, bias=False)
#        print(self.linear.weight.data.shape)

    def forward(self, Z):
        # orthogonal projection (constraint)
        U, D, V = Z.svd() # singular value decomposition of representation Z
        D = torch.diag(1/D) # create diagonal matrix from inverse singular value
        new_W = V[:, :self.input_dim].matmul(D[:self.input_dim, :self.output_dim])
#        print(new_W.shape)
        self.linear.weight = nn.Parameter(new_W.t()) #assign the new weight
#        print(self.linear.weight.data.shape)
#        self.constrain(x)
        return self.linear(Z)
    

class VAEOrtho(nn.Module):
    '''
    VAE with orthogonal projection
    '''
    def __init__(self,configs):
        super(VAEOrtho, self).__init__()
        self.encoder = Encoder(configs['encoder'])
        self.mid =NNNorm(**configs['mid'])
        self.ortho = OrthogonalProjector(**configs['ortho'])
        self.decoder = Decoder(configs['decoder'])

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(z.size(0), -1)
        z = self.mid(z)
        z = self.ortho(z)      # <--- orthogonal projection
        x_ = z.view(z.size(0), -1, 4, 4)
        x_ = self.decoder(x_)
        return x_, z

