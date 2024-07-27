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

class NNNormRelu(nn.Sequential):
    def __init__(self, in_features, out_features):
        super(NNNorm, self).__init__(
            nn.Linear(in_features, out_features, bias=False),
            nn.BatchNorm1d(out_features, eps=1e-4, affine=False),
            nn.LeakyReLU()
        )
class NNNorm(nn.Sequential):
    def __init__(self, in_features, out_features):
        super(NNNorm, self).__init__(
            nn.Linear(in_features, out_features, bias=False),
            nn.BatchNorm1d(out_features, eps=1e-4, affine=False)
        )

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode = 'nearest'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor , mode=self.mode)
        return x
    
class Elu(nn.Module):
    def __init__(self):
        super(Elu, self).__init__()
        self.elu = nn.functional.elu

    def forward(self, x):
        x = self.elu(x)
        return x


class DeconvNet(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding, scale):
        super(DeconvNet, self).__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding = padding, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-4, affine=False),
            nn.ELU(), #Elu(),#nn.ELU(),#nn.LeakyReLU(),
            nn.Upsample(scale_factor=scale, mode='nearest')
#            Interpolate(scale)
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
    

class Orthogonal_Projector_try(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
        )

    def forward(self, x):
        return self.block(x)


class weightConstraint(object):
    def __init__(self, input):
        self.input = input
        # pass

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data.t()
            d = w.shape[1]
            U, S, V = self.input.svd()
            S = S.diag()
            W = V[:, :d].matmul(S[:d, :d])
            module.weight.data = W.t()

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
#        self.ortho = Orthogonal_Projector_try(**configs['ortho'])
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
#        self.ortho.apply(weightConstraint(z))      # <--- orthogonal projection
        z = self.ortho(z)      # <--- orthogonal projection
        x_ = z.view(z.size(0), -1, 4, 4)
        x_ = self.relu(x_)
#        x_ = z.view(z.size(0), -1, 4, 4)
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
#        self.ortho = Orthogonal_Projector_try(**configs['ortho'])

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(z.size(0), -1)
        z = self.mid(z)
#        self.ortho.apply(weightConstraint(z))
        z = self.ortho(z)      # <--- orthogonal projection  
        return z
    

'''
original
[ 7.8339,  0.4046, -0.5191, -0.2870, -0.1000,  0.1213, -0.1000,  0.1000,
         0.1231,  0.1000,  0.1000, -0.1000, -0.1000, -0.1000, -0.1000,  0.1000,
        -0.1000, -0.1000, -0.1000,  0.1000, -0.1000, -0.1000, -0.1000,  0.1000,
        -0.1000,  0.1000,  0.1000, -0.1000, -0.1000,  0.1000, -0.1000,  0.1000]

[-1.0000,  1.3861,  0.5671,  0.2030,  0.1000,  0.1000,  0.1000,  0.1000,
         0.1000,  0.1000,  0.1000,  0.1000,  0.1000,  0.1000,  0.1000,  0.1000,
         0.1000,  0.1000,  0.1000,  0.1000, -0.1000,  0.1000, -0.1000, -0.1000,
         0.1000,  0.1000,  0.1000,  0.1000, -0.1000,  0.1000, -0.1000,  0.1000]

mine with xavier_uniform
[-0.9977,  1.1554,  0.1000,  0.1000,  0.1000,  0.1000,  0.1000,  0.1000,
         0.1000,  0.1000,  0.1000,  0.1000,  0.1000,  0.1000,  0.1000,  0.1000,
        -0.1000,  0.1000,  0.1000, -0.1000,  0.1000,  0.1000, -0.1000, -0.1000,
        -0.1000, -0.1000,  0.1000,  0.1000, -0.1000, -0.1000,  0.1000, -0.1000])


tensor([ 1.4172e+02,  2.4041e+00,  8.0659e-01,  3.1899e-01,  2.1644e-01,
         1.0000e-01,  1.0000e-01,  1.0000e-01,  1.0000e-01,  1.0000e-01,
         1.0000e-01,  1.0000e-01,  1.0000e-01,  1.0000e-01,  1.0000e-01,
         1.0000e-01,  1.0000e-01,  1.0000e-01,  1.0000e-01, -1.0000e-01,
         1.0000e-01,  1.0000e-01,  1.0000e-01, -1.0000e-01,  1.0000e-01,
         1.0000e-01,  1.0000e-01,  1.0000e-01,  1.0000e-01,  1.0000e-01,
         1.0000e-01,  1.0000e-01])
'''

import torch.nn.functional as F

class pretrain_autoencoder_cifar(nn.Module):

    def __init__(self, rep_dim=128):
        super(pretrain_autoencoder_cifar, self).__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)
        self.orthogonal_projector = Orthogonal_Projector_try(self.rep_dim)
        self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight)
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight)
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight)
        self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.elu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.elu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.elu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        x = self.bn1d(self.fc1(x))
        self.orthogonal_projector.apply(weightConstraint(x))
        middle = self.orthogonal_projector(x)
        x = middle.view(middle.size(0), int(self.rep_dim / (4 * 4)), 4, 4)
        x = F.elu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.elu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.elu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.elu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv4(x)
        x = torch.sigmoid(x)
        return x, middle