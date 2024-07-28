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
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-4, affine=False),
            nn.ELU(), #Elu(),#nn.ELU(),#nn.LeakyReLU(),
            nn.MaxPool2d(pool_size)
        )

class NNNorm(nn.Sequential):
    def __init__(self, in_features, out_features):
        super(NNNorm, self).__init__(
            nn.Linear(in_features, out_features, bias=False),
            nn.BatchNorm1d(out_features, eps=1e-4, affine=False)
        )


class DeconvNet(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding, scale):
        super(DeconvNet, self).__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding = padding, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-4, affine=False),
            nn.ELU(), #nn.LeakyReLU(),
            nn.Upsample(scale_factor=scale, mode='nearest')
        )

class DeconvNet2(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DeconvNet2, self).__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding = padding, bias=False),
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

class AE(nn.Module):
    '''
    AE stands for Autoencoder which consists of encoder and decoder
    Its task is to reconstruct the input data by training
    A signature of the autoencoder is to have a bottleneck in the middle
    We assume the latent representation z to be in the middle (the output of the encoder)
    '''
    def __init__(self,configs):
        super(AE, self).__init__()
        self.encoder = Encoder(configs['encoder'])
        self.mid =NNNorm(**configs['mid'])
        self.relu = nn.LeakyReLU()
        self.decoder = Decoder(configs['decoder'])
        # initialize weights convolutional layers
        self.encoder.apply(init_weights_conv)
        self.decoder.apply(init_weights_conv2)

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(z.size(0), -1)
        z = self.mid(z)
        x_ = self.relu(z)
        x_ = x_.view(x_.size(0), -1, 4, 4) # 4 is fixed here to create a compressed representation
        x_ = self.decoder(x_)
        return x_, z
    
class AEskip(nn.Module):
    '''
    AE with skip connection
    '''
    def __init__(self, configs):
        super(AEskip, self).__init__()
        self.encoder = Encoder(configs['encoder'])
        self.mid =NNNorm(**configs['mid'])
        self.relu = nn.LeakyReLU()
        self.decoder = Decoder(configs['decoder'])

    def forward(self, x):
        x0 = x
        z = self.encoder(x)
        z = z.view(z.size(0), -1)
        z = self.mid(z)
        z = self.relu(z)
        z = z.view(z.size(0), -1, 4, 4)
        x_ = self.decoder(z)
        x_ = x_ + x0 #skip connection
        return x_, z

class Enc(nn.Module):
    '''
    Encoder only
    '''
    def __init__(self,configs):
        super(Enc, self).__init__()
        self.encoder = Encoder(configs['encoder'])
        self.mid =NNNorm(**configs['mid'])

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(z.size(0), -1)
        z = self.mid(z)
        return z

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
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, Z):
        # orthogonal projection (constraint)
        k_new = self.linear.weight.data.shape[0]
        U, D, V = Z.svd() # singular value decomposition of representation Z
        D = torch.diag(D)#D.diag() #torch.diag(1/D) # create diagonal matrix from inverse singular value
        new_W = V[:, :k_new].matmul(D[:k_new, :k_new]) # new weight
#        print(new_W.shape)
        self.linear.weight.data = new_W.t()  #assign the new weight
        return self.linear(Z)

def init_weights_conv(m): #initialize weights
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)

def init_weights_mlp(m): #initialize weights
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

def init_weights_conv2(m): #initialize weights
    if isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight)

def init_weights_normal(m): #initialize weights
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

class AEOrtho(nn.Module):
    '''
    VAE with orthogonal projection
    '''
    def __init__(self,configs):
        super(AEOrtho, self).__init__()
        self.encoder = Encoder(configs['encoder'])
        self.mid =NNNorm(**configs['mid'])
        self.relu = nn.ELU() #Elu()#nn.ELU() #nn.LeakyReLU()
        self.ortho = OrthogonalProjector(**configs['ortho'])
        self.decoder = Decoder(configs['decoder'])
        # initialize weights convolutional layers
        self.encoder.apply(init_weights_conv)
        self.mid.apply(init_weights_mlp)
        self.ortho.apply(init_weights_normal)
        self.decoder.apply(init_weights_conv2)

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(z.size(0), -1)
        z = self.mid(z)
        z = self.ortho(z)      # <--- orthogonal projection
        x_ = z.view(z.size(0), -1, 4, 4)
        x_ = self.relu(x_)
        x_ = self.decoder(x_)
        return x_, z

class AEskipOrtho(nn.Module):
    '''
    VAE with orthogonal projection and skip connection
    '''
    def __init__(self,configs):
        super(AEskipOrtho, self).__init__()
        self.encoder = Encoder(configs['encoder'])
        self.mid =NNNorm(**configs['mid'])
        self.relu = nn.LeakyReLU()
        self.ortho = OrthogonalProjector(**configs['ortho'])
        self.decoder = Decoder(configs['decoder'])

    def forward(self, x):
        x0 = x
        z = self.encoder(x)
        z = z.view(z.size(0), -1)
        z = self.mid(z)
        z = self.ortho(z)      # <--- orthogonal projection
        x_ = z.view(z.size(0), -1, 4, 4)        
        x_ = self.relu(x_)
        x_ = self.decoder(x_)
        x_ = x_ + x0 #skip connection
        return x_, z

class EncOrtho(nn.Module):
    '''
    Encoder with orthogonal projection
    '''
    def __init__(self,configs):
        super(EncOrtho, self).__init__()
        self.encoder = Encoder(configs['encoder'])
        self.mid =NNNorm(**configs['mid'])
        self.ortho = OrthogonalProjector(**configs['ortho'])

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(z.size(0), -1)
        z = self.mid(z)
        z = self.ortho(z)      # <--- orthogonal projection  
        return z
    