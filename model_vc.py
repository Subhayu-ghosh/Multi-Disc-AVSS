from pathlib import Path

import imageio
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

from scipy.fftpack import dct
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer
from audioUtils.audio import wav2seg, inv_preemphasis, preemphasis
from data.Sample_dataset import pad_seq
from model_video import VideoGenerator, STAGE2_G
from saveWav import mel2wav
from audioUtils.hparams import hparams
from audioUtils import audio
from vocoder.models.fatchord_version import WaveRNN
import cv2

_inv_mel_basis = np.linalg.pinv(audio._build_mel_basis(hparams))
mel_basis = librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=40)

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class MyEncoder(nn.Module):
    '''Encoder without speaker embedding'''

    def __init__(self, dim_neck, freq, num_mel=80):
        super(MyEncoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq

        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(num_mel if i == 0 else 512,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, x, return_unsample=False):
        # (B, T, n_mel)
        x = x.squeeze(1).transpose(2, 1)

        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]

        # print(outputs.shape)

        codes = []
        for i in range(0, outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:, i + self.freq - 1, :], out_backward[:, i, :]), dim=-1))
        if return_unsample:
            return codes, outputs
        return codes


class Decoder(nn.Module):
    """Decoder module:
    """
    def __init__(self, dim_neck, dim_emb, dim_pre, num_mel=80):
        super(Decoder, self).__init__()
        
        self.lstm1 = nn.LSTM(dim_neck*2+dim_emb, dim_pre, 1, batch_first=True)
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre,
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)
        
        self.linear_projection = LinearNorm(1024, num_mel)

    def forward(self, x):
        
        #self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        outputs, _ = self.lstm2(x)
        
        decoder_output = self.linear_projection(outputs)

        return decoder_output   

    
class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, num_mel=80):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(num_mel, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, num_mel,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(num_mel))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


def pad_layer(inp, layer, is_2d=False):
    if type(layer.kernel_size) == tuple:
        kernel_size = layer.kernel_size[0]
    else:
        kernel_size = layer.kernel_size
    if not is_2d:
        if kernel_size % 2 == 0:
            pad = (kernel_size//2, kernel_size//2 - 1)
        else:
            pad = (kernel_size//2, kernel_size//2)
    else:
        if kernel_size % 2 == 0:
            pad = (kernel_size//2, kernel_size//2 - 1, kernel_size//2, kernel_size//2 - 1)
        else:
            pad = (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2)
    # padding
    inp = F.pad(inp,
            pad=pad,
            mode='reflect')
    out = layer(inp)
    return out

class UInet(nn.Module):
    def __init__(self, embedding_user, embedding_item, embedding_size=16, out_channels=64, kernel_size=2, stride=1, padding=0, n_class=1):
        super(UInet, self).__init__()
        self.embedding_size = embedding_size
        self.embedding_user, self.embedding_item = embedding_user, embedding_item
        self.cnn = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(int(((self.embedding_size-kernel_size+2*padding)/stride+1)*out_channels), n_class)
 
    def forward(self, x):
        embed_users = self.embedding_user(x[:,0])
        embed_items0 = self.embedding_item(x[:,1])
        embed_items1 = self.embedding_item(x[:,2])
        out = torch.cat([embed_users, embed_items0],1).reshape(-1, 1, 2, self.embedding_size)
        out = self.cnn(out)          
        out = self.relu(out)
        out = torch.flatten(out, 1)
        out = self.linear(out) 
        return out
    
    def predict(self, pairs, batch_size, verbose):
        """Computes predictions for a given set of user-item pairs.
        Args:
          pairs: A pair of lists (users, items) of the same length.
          batch_size: unused.
          verbose: unused.
        Returns:
          predictions: A list of the same length as users and items, such that
          predictions[i] is the models prediction for (users[i], items[i]).
        """
        del batch_size, verbose
        num_examples = len(pairs[0])
        assert num_examples == len(pairs[1])
        predictions = np.empty(num_examples)
        pairs = np.array(pairs, dtype=np.int16)
        for i in range(num_examples):
            x = np.c_[pairs[0][i],pairs[1][i],pairs[1][i]]
            x = torch.from_numpy(x).long()
            out = self.forward(x)
            predictions[i] = out.reshape(-1).data.numpy()
        return predictions

class UIInet(nn.Module):
    def __init__(self, embedding_user, embedding_item, embedding_size=16, out_channels=64, kernel_size=2, stride=1, padding=0, n_class=1):
        super(UIInet, self).__init__()
        self.embedding_size, self.kernel_size = embedding_size, kernel_size
        self.embedding_user, self.embedding_item = embedding_user, embedding_item
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.cnn2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        if self.kernel_size == 2:
            input_size = (self.embedding_size - self.kernel_size + 2 * padding)/stride + 1
            channel_num = out_channels
        else:
            input_size = self.embedding_size
            channel_num = out_channels
        self.linear = nn.Linear(int(((input_size - self.kernel_size + 2 * padding)/stride + 1) * channel_num), n_class)
 
    def forward(self, x):
        embed_users = self.embedding_user(x[:,0])
        embed_items0 = self.embedding_item(x[:,1])
        embed_items1 = self.embedding_item(x[:,2])
        out = torch.cat([embed_items0, embed_users, embed_items1],1).reshape(-1, 1, 3, self.embedding_size)
        out = self.cnn1(out)          
        out = self.relu(out)
        if self.kernel_size == 2: 
            out = self.cnn2(out)          
            out = self.relu(out)
        out = torch.flatten(out, 1)
        out = self.linear(out) 
        return out
    
    def predict(self, pairs, batch_size, verbose):
        """Computes predictions for a given set of user-item pairs.
        Args:
          pairs: A pair of lists (users, items) of the same length.
          batch_size: unused.
          verbose: unused.
        Returns:
          predictions: A list of the same length as users and items, such that
          predictions[i] is the models prediction for (users[i], items[i]).
        """
        del batch_size, verbose
        num_examples = len(pairs[0])
        assert num_examples == len(pairs[1])
        predictions = np.empty(num_examples)
        pairs = np.array(pairs, dtype=np.int16)
        for i in range(num_examples):
            x = np.c_[pairs[0][i],pairs[1][i],pairs[1][i]]
            x = torch.from_numpy(x).long()
            out = self.forward(x)
            predictions[i] = out.reshape(-1).data.numpy()
        return predictions

class PatchDiscriminator1(nn.Module):
    def __init__(self, users_num, items_num, embedding_size=16, out_channels=64, kernel_size=2, stride=1, padding=0, n_class=1):
        super(Net, self).__init__()
        self.embedding_size, self.kernel_size, self.items_num, self.users_num = embedding_size, kernel_size, items_num, users_num
        self.embedding_user  = nn.Embedding(self.users_num, self.embedding_size)
        self.embedding_item = nn.Embedding(self.items_num, self.embedding_size)
        #self.embedding_user  = nn.Embedding.from_pretrained(torch.nn.init.normal(tensor=torch.Tensor(self.users_num, self.embedding_size), mean=0, std=0.1))
        #self.embedding_item = nn.Embedding.from_pretrained(torch.nn.init.normal(tensor=torch.Tensor(self.items_num, self.embedding_size), mean=0, std=0.1))

        self.net_ui = UInet(embedding_user=self.embedding_user, 
                            embedding_item=self.embedding_item, 
                            embedding_size=self.embedding_size, 
                            out_channels=out_channels, 
                            kernel_size=2, 
                            stride=stride, 
                            padding=padding, 
                            n_class=n_class)
        self.net_uii = UIInet(embedding_user=self.embedding_user, 
                              embedding_item=self.embedding_item, 
                              embedding_size=self.embedding_size, 
                              out_channels=out_channels, 
                              kernel_size=self.kernel_size, 
                              stride=stride, 
                              padding=padding, 
                              n_class=n_class)
 
    def forward(self, x):
        out1 = self.net_ui(x)          
        out2 = self.net_uii(x) 
        return out1, out2
    
    def predict(self, pairs, batch_size, verbose):
        """Computes predictions for a given set of user-item pairs.
        Args:
          pairs: A pair of lists (users, items) of the same length.
          batch_size: unused.
          verbose: unused.
        Returns:
          predictions: A list of the same length as users and items, such that
          predictions[i] is the models prediction for (users[i], items[i]).
        """
        del batch_size, verbose
        num_examples = len(pairs[0])
        assert num_examples == len(pairs[1])
        predictions = np.empty(num_examples)
        pairs = np.array(pairs, dtype=np.int16)
        for i in range(num_examples):
            x = np.c_[pairs[0][i],pairs[1][i],pairs[1][i]]
            x = torch.from_numpy(x).long()
            out, _ = self.forward(x)
            predictions[i] = out.reshape(-1).data.numpy()
        return predictions
    
    def get_embeddings(self):
        idx = torch.LongTensor([i for i in range(self.items_num)])
        embeddings = self.embedding_item(idx)
        return embeddings

class ConvLayer(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=256):
        '''Constructs the ConvLayer with a specified input and output size.
           param in_channels: input depth of an image, default value = 1
           param out_channels: output depth of the convolutional layer, default value = 256
           '''
        super(ConvLayer, self).__init__()

        # defining a convolutional layer of the specified size
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=9, stride=1, padding=0)

    def forward(self, x):
        '''Defines the feedforward behavior.
           param x: the input to the layer; an input image
           return: a relu-activated, convolutional layer
           '''
        # applying a ReLu activation to the outputs of the conv layer
        features = F.relu(self.conv(x)) # will have dimensions (batch_size, 20, 20, 256)
        return features

class PrimaryCaps(nn.Module):
    
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32):
        '''Constructs a list of convolutional layers to be used in 
           creating capsule output vectors.
           param num_capsules: number of capsules to create
           param in_channels: input depth of features, default value = 256
           param out_channels: output depth of the convolutional layers, default value = 32
           '''
        super(PrimaryCaps, self).__init__()

        # creating a list of convolutional layers for each capsule I want to create
        # all capsules have a conv layer with the same parameters
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=9, stride=2, padding=0)
            for _ in range(num_capsules)])
    
    def forward(self, x):
        '''Defines the feedforward behavior.
           param x: the input; features from a convolutional layer
           return: a set of normalized, capsule output vectors
           '''
        # get batch size of inputs
        batch_size = x.size(0)
        # reshape convolutional layer outputs to be (batch_size, vector_dim=1152, 1)
        u = [capsule(x).view(batch_size, 32 * 6 * 6, 1) for capsule in self.capsules]
        # stack up output vectors, u, one for each capsule
        u = torch.cat(u, dim=-1)
        # squashing the stack of vectors
        u_squash = self.squash(u)
        return u_squash
    def squash(self, input_tensor):
        '''Squashes an input Tensor so it has a magnitude between 0-1.
           param input_tensor: a stack of capsule inputs, s_j
           return: a stack of normalized, capsule output vectors, v_j
           '''
        squared_norm = (input_tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm) # normalization coeff
        output_tensor = scale * input_tensor / torch.sqrt(squared_norm)    
        return output_tensor    

# to get transpose softmax function

# dynamic routing
def dynamic_routing(b_ij, u_hat, squash, routing_iterations=3):
    '''Performs dynamic routing between two capsule layers.
       param b_ij: initial log probabilities that capsule i should be coupled to capsule j
       param u_hat: input, weighted capsule vectors, W u
       param squash: given, normalizing squash function
       param routing_iterations: number of times to update coupling coefficients
       return: v_j, output capsule vectors
       '''    
    # update b_ij, c_ij for number of routing iterations
    for iteration in range(routing_iterations):
        # softmax calculation of coupling coefficients, c_ij
        c_ij = softmax(b_ij, dim=2)

        # calculating total capsule inputs, s_j = sum(c_ij*u_hat)
        s_j = (c_ij * u_hat).sum(dim=2, keepdim=True)

        # squashing to get a normalized vector output, v_j
        v_j = squash(s_j)

        # if not on the last iteration, calculate agreement and new b_ij
        if iteration < routing_iterations - 1:
            # agreement
            a_ij = (u_hat * v_j).sum(dim=-1, keepdim=True)
            
            # new b_ij
            b_ij = b_ij + a_ij
    
    return v_j # return latest v_j

class DigitCaps(nn.Module):
    
    def __init__(self, num_capsules=10, previous_layer_nodes=32*6*6, 
                 in_channels=8, out_channels=16):
        '''Constructs an initial weight matrix, W, and sets class variables.
           param num_capsules: number of capsules to create
           param previous_layer_nodes: dimension of input capsule vector, default value = 1152
           param in_channels: number of capsules in previous layer, default value = 8
           param out_channels: dimensions of output capsule vector, default value = 16
           '''
        super(DigitCaps, self).__init__()

        # setting class variables
        self.num_capsules = num_capsules
        self.previous_layer_nodes = previous_layer_nodes # vector input (dim=1152)
        self.in_channels = in_channels # previous layer's number of capsules

        # starting out with a randomly initialized weight matrix, W
        # these will be the weights connecting the PrimaryCaps and DigitCaps layers
        self.W = nn.Parameter(torch.randn(num_capsules, previous_layer_nodes, 
                                          in_channels, out_channels))

    def forward(self, u):
        '''Defines the feedforward behavior.
           param u: the input; vectors from the previous PrimaryCaps layer
           return: a set of normalized, capsule output vectors
           '''
        
        # adding batch_size dims and stacking all u vectors
        u = u[None, :, :, None, :]
        # 4D weight matrix
        W = self.W[:, None, :, :, :]
        
        # calculating u_hat = W*u
        u_hat = torch.matmul(u, W)
        # getting the correct size of b_ij
        # setting them all to 0, initially
        b_ij = torch.zeros(*u_hat.size())
        
        # moving b_ij to GPU, if available
        if TRAIN_ON_GPU:
            b_ij = b_ij.cuda()

        # update coupling coefficients and calculate v_j
        v_j = dynamic_routing(b_ij, u_hat, self.squash, routing_iterations=3)

        return v_j # return final vector outputs

    def squash(self, input_tensor):
        '''Squashes an input Tensor so it has a magnitude between 0-1.
           param input_tensor: a stack of capsule inputs, s_j
           return: a stack of normalized, capsule output vectors, v_j
           '''
        # same squash function as before
        squared_norm = (input_tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm) # normalization coeff
        output_tensor = scale * input_tensor / torch.sqrt(squared_norm)    
        return output_tensor

class Decodar(nn.Module):
    
    def __init__(self, input_vector_length=16, input_capsules=10, hidden_dim=512):
        '''Constructs an series of linear layers + activations.
           param input_vector_length: dimension of input capsule vector, default value = 16
           param input_capsules: number of capsules in previous layer, default value = 10
           param hidden_dim: dimensions of hidden layers, default value = 512
           '''
        super(Decodar, self).__init__()
        
        # calculate input_dim
        input_dim = input_vector_length * input_capsules
        
        # define linear layers + activations
        self.linear_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # first hidden layer
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim*2), # second, twice as deep
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim*2, 28*28), # can be reshaped into 28*28 image
            nn.Sigmoid() # sigmoid activation to get output pixel values in a range from 0-1
            )
        
    def forward(self, x):
        '''Defines the feedforward behavior.
           param x: the input; vectors from the previous DigitCaps layer
           return: two things, reconstructed images and the class scores, y
           '''
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)
        
        # find the capsule with the maximum vector length
        # here, vector length indicates the probability of a class' existence
        _, max_length_indices = classes.max(dim=1)
        
        # create a sparse class matrix
        sparse_matrix = torch.eye(10) # 10 is the number of classes
        if TRAIN_ON_GPU:
            sparse_matrix = sparse_matrix.cuda()
        # get the class scores from the "correct" capsule
        y = sparse_matrix.index_select(dim=0, index=max_length_indices.data)
        
        # create reconstructed pixels
        x = x * y[:, :, None]
        # flatten image into a vector shape (batch_size, vector_dim)
        flattened_x = x.contiguous().view(x.size(0), -1)
        # create reconstructed image vectors
        reconstructions = self.linear_layers(flattened_x)
        
        # return reconstructions and the class scores, y
        return reconstructions, y


class PatchDiscriminator2(nn.Module):
    
    def __init__(self):
        '''Constructs a complete Capsule Network.'''
        super(CapsuleNetwork, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps()
        self.decodar = Decodar()
                
    def forward(self, images):
        '''Defines the feedforward behavior.
           param images: the original MNIST image input data
           return: output of DigitCaps layer, reconstructed images, class scores
           '''
        primary_caps_output = self.primary_capsules(self.conv_layer(images))
        caps_output = self.digit_capsules(primary_caps_output).squeeze().transpose(0,1)
        reconstructions, y = self.decodar(caps_output)
        return caps_output, reconstructions, y

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class PatchDiscriminator3(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq, dim_spec=80, is_train=False, lr=0.001, loss_content=True,
                 discriminator=False, multigpu=False, lambda_gan=0.0001,
                 lambda_wavenet=0.001, args=None,
                 test_path=None):
        super(Generator, self).__init__()

        self.encoder = MyEncoder(dim_neck, freq, num_mel=dim_spec)
        self.decoder = Decoder(dim_neck, 0, dim_pre, num_mel=dim_spec)
        self.postnet = Postnet(num_mel=dim_spec)

        if discriminator:
            self.dis1 = PatchDiscriminator1(n_class=num_speakers)
            self.dis2 = PatchDiscriminator2(n_class=num_speakers)
            self.dis3 = PatchDiscriminator3(n_class=num_speakers)
            self.dis1_criterion = GANLoss(use_lsgan=use_lsgan, tensor=torch.cuda.FloatTensor)
            self.dis2_criterion = GANLoss(use_lsgan=use_lsgan, tensor=torch.cuda.FloatTensor)
            self.dis3_criterion = GANLoss(use_lsgan=use_lsgan, tensor=torch.cuda.FloatTensor)
        else:
            self.dis = None

        self.loss_content = loss_content
        self.lambda_gan = lambda_gan
        self.lambda_wavenet = lambda_wavenet

        self.multigpu = multigpu
        if test_path is not None:
            self.prepare_test(dim_spec, test_path)

        self.vocoder = WaveRNN(
            rnn_dims=hparams.voc_rnn_dims,
            fc_dims=hparams.voc_fc_dims,
            bits=hparams.bits,
            pad=hparams.voc_pad,
            upsample_factors=hparams.voc_upsample_factors,
            feat_dims=hparams.num_mels,
            compute_dims=hparams.voc_compute_dims,
            res_out_dims=hparams.voc_res_out_dims,
            res_blocks=hparams.voc_res_blocks,
            hop_length=hparams.hop_size,
            sample_rate=hparams.sample_rate,
            mode=hparams.voc_mode
        )
        
        if is_train:
            self.criterionIdt = torch.nn.L1Loss(reduction='mean')
            self.opt_encoder = torch.optim.Adam(self.encoder.parameters(), lr=lr)
            self.opt_decoder = torch.optim.Adam(itertools.chain(self.decoder.parameters(), self.postnet.parameters()), lr=lr)
            if discriminator:
                self.opt_dis = torch.optim.Adam(self.dis.parameters(), lr=lr)
            self.opt_vocoder = torch.optim.Adam(self.vocoder.parameters(), lr=hparams.voc_lr)
            self.vocoder_loss_func = F.cross_entropy # Only for RAW


        if multigpu:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)
            self.postnet = nn.DataParallel(self.postnet)
            self.vocoder = nn.DataParallel(self.vocoder)
            if self.dis is not None:
                self.dis = nn.DataParallel(self.dis)

    def prepare_test(self, dim_spec, test_path):
        mel_basis80 = librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=80)

        wav, sr = librosa.load(test_path, hparams.sample_rate)
        wav = preemphasis(wav, hparams.preemphasis, hparams.preemphasize)
        linear_spec = np.abs(
            librosa.stft(wav, n_fft=hparams.n_fft, hop_length=hparams.hop_size, win_length=hparams.win_size))
        mel_spec = mel_basis80.dot(linear_spec)
        mel_db = 20 * np.log10(mel_spec)
        source_spec = np.clip((mel_db + 120) / 125, 0, 1)

        self.test_wav = wav

        self.test_spec = torch.Tensor(pad_seq(source_spec.T, hparams.freq)).unsqueeze(0)

    def test_fixed(self, device):
        with torch.no_grad():
            s2t_spec = self.conversion(self.test_spec, device).cpu()

        ret_dic = {}
        ret_dic['A_fake_griffin'], sr = mel2wav(s2t_spec.numpy().squeeze(0).T)
        ret_dic['A'] = self.test_wav

        with torch.no_grad():
            if not self.multigpu:
                ret_dic['A_fake_w'] = inv_preemphasis(self.vocoder.generate(s2t_spec.to(device).transpose(2, 1), False, None, None, mu_law=True),
                                                hparams.preemphasis, hparams.preemphasize)
            else:
                ret_dic['A_fake_w'] = inv_preemphasis(self.vocoder.module.generate(s2t_spec.to(device).transpose(2, 1), False, None, None, mu_law=True),
                                                hparams.preemphasis, hparams.preemphasize)
        return ret_dic, sr


    def conversion(self, spec, device, speed=1):
        spec = spec.to(device)
        if not self.multigpu:
            codes = self.encoder(spec)
        else:
            codes = self.encoder.module(spec)
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1, int(speed * spec.size(1) / len(codes)), -1))
        code_exp = torch.cat(tmp, dim=1)
        mel_outputs = self.decoder(code_exp) if not self.multigpu else self.decoder.module(code_exp)

        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2, 1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2, 1)
        return mel_outputs_postnet

    def optimize_parameters(self, dataloader, epochs, device, display_freq=10, save_freq=1000, save_dir="./",
                            experimentName="Train", load_model=None, initial_niter=0):
        writer = SummaryWriter(log_dir="logs/"+experimentName)
        if load_model is not None:
            print("Loading from %s..." % load_model)
            # self.load_state_dict(torch.load(load_model))
            d = torch.load(load_model)
            newdict = d.copy()
            for key, value in d.items():
                newkey = key
                if 'wavenet' in key:
                    newdict[key.replace('wavenet', 'vocoder')] = newdict.pop(key)
                    newkey = key.replace('wavenet', 'vocoder')
                if self.multigpu and 'module' not in key:
                    newdict[newkey.replace('.','.module.',1)] = newdict.pop(newkey)
                    newkey = newkey.replace('.', '.module.', 1)
                if newkey not in self.state_dict():
                    newdict.pop(newkey)
            self.load_state_dict(newdict)
            print("AutoVC Model Loaded")
        niter = initial_niter
        for epoch in range(epochs):
            self.train()
            for i, data in enumerate(dataloader):
                speaker_org, spec, prev, wav = data
                loss_dict, loss_dict_discriminator, loss_dict_wavenet = \
                    self.train_step(spec.to(device), speaker_org.to(device), prev=prev.to(device), wav=wav.to(device), device=device)
                if niter % display_freq == 0:
                    print("Epoch[%d] Iter[%d] Niter[%d] %s %s %s"
                          % (epoch, i, niter, loss_dict, loss_dict_discriminator, loss_dict_wavenet))
                    writer.add_scalars('data/Loss', loss_dict,
                                       niter)
                    if loss_dict_discriminator != {}:
                        writer.add_scalars('data/discriminator', loss_dict_discriminator, niter)
                    if loss_dict_wavenet != {}:
                        writer.add_scalars('data/wavenet', loss_dict_wavenet, niter)
                if niter % save_freq == 0:
                    print("Saving and Testing...", end='\t')
                    torch.save(self.state_dict(), save_dir + '/Epoch' + str(epoch).zfill(3) + '_Iter'
                               + str(niter).zfill(8) + ".pkl")
                    # self.load_state_dict(torch.load('params.pkl'))
                    if len(dataloader) >= 2 and self.test_wav is not None:
                        wav_dic, sr = self.test_fixed(device)
                        for key, wav in wav_dic.items():
                            # print(wav.shape)
                            writer.add_audio(key, wav, niter, sample_rate=sr)
                        librosa.output.write_wav(save_dir + '/Iter' + str(niter).zfill(8) +'.wav', wav_dic['A_fake_w'].astype(np.float32), hparams.sample_rate)
                    print("Done")
                    self.train()
                torch.cuda.empty_cache()  # Prevent Out of Memory
                niter += 1


    def train_step(self, x, c_org, mask=None, mask_code=None, prev=None, wav=None,
                   ret_content=False, retain_graph=False, device='cuda:0'):
        codes = self.encoder(x)
        # print(codes[0].shape)
        content = torch.cat([code.unsqueeze(1) for code in codes], dim=1)
        # print("content shape", content.shape)
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1, int(x.size(1) / len(codes)), -1))
        code_exp = torch.cat(tmp, dim=1)

        encoder_outputs = torch.cat((code_exp, c_org.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1)

        mel_outputs = self.decoder(code_exp)

        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2, 1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2, 1)

        loss_dict, loss_dict_discriminator, loss_dict_wavenet = {}, {}, {}

        loss_recon = self.criterionIdt(x, mel_outputs)
        loss_recon0 = self.criterionIdt(x, mel_outputs_postnet)
        loss_dict['recon'], loss_dict['recon0'] = loss_recon.data.item(), loss_recon0.data.item()

        if self.loss_content:
            recons_codes = self.encoder(mel_outputs_postnet)
            recons_content = torch.cat([code.unsqueeze(1) for code in recons_codes], dim=1)
            if mask is not None:
                loss_content = self.criterionIdt(content.masked_select(mask_code.byte()), recons_content.masked_select(mask_code.byte()))
            else:
                loss_content = self.criterionIdt(content, recons_content)
            loss_dict['content'] = loss_content.data.item()
        else:
            loss_content = torch.from_numpy(np.array(0))

        loss_gen, loss_dis, loss_vocoder = [torch.from_numpy(np.array(0))] * 3
        fake_mel = None
        if self.dis1:
            # true_label = torch.from_numpy(np.ones(shape=(x.shape[0]))).to('cuda:0').long()
            # false_label = torch.from_numpy(np.zeros(shape=(x.shape[0]))).to('cuda:0').long()

            fake_mel = self.conversion(x, device)

            loss_dis = self.dis1_criterion(self.dis1(x), True) + self.dis1_criterion(self.dis(fake_mel), False)
                       # +  self.dis_criterion(self.dis(mel_outputs_postnet), False)

            self.opt_dis.zero_grad()
            loss_dis.backward(retain_graph=True)
            self.opt_dis.step()
            loss_gen = self.dis1_criterion(self.dis1(fake_mel), True)
                # + self.dis_criterion(self.dis(mel_outputs_postnet), True)
            loss_dict_discriminator['dis1'], loss_dict_discriminator['gen'] = loss_dis.data.item(), loss_gen.data.item()

        elif self.dis2:
            # true_label = torch.from_numpy(np.ones(shape=(x.shape[0]))).to('cuda:0').long()
            # false_label = torch.from_numpy(np.zeros(shape=(x.shape[0]))).to('cuda:0').long()

            fake_mel = self.conversion(x, device)

            loss_dis = self.dis2_criterion(self.dis2(x), True) + self.dis2_criterion(self.dis2(fake_mel), False)
                       # +  self.dis_criterion(self.dis(mel_outputs_postnet), False)

            self.opt_dis.zero_grad()
            loss_dis.backward(retain_graph=True)
            self.opt_dis.step()
            loss_gen = self.dis2_criterion(self.dis2(fake_mel), True)
                # + self.dis_criterion(self.dis(mel_outputs_postnet), True)
            loss_dict_discriminator['dis2'], loss_dict_discriminator['gen'] = loss_dis.data.item(), loss_gen.data.item() 
            
        elif self.dis3:
            # true_label = torch.from_numpy(np.ones(shape=(x.shape[0]))).to('cuda:0').long()
            # false_label = torch.from_numpy(np.zeros(shape=(x.shape[0]))).to('cuda:0').long()

            fake_mel = self.conversion(x, device)

            loss_dis = self.dis3_criterion(self.dis3(x), True) + self.dis3_criterion(self.dis3(fake_mel), False)
                       # +  self.dis_criterion(self.dis(mel_outputs_postnet), False)

            self.opt_dis.zero_grad()
            loss_dis.backward(retain_graph=True)
            self.opt_dis.step()
            loss_gen = self.dis3_criterion(self.dis2(fake_mel), True)
                # + self.dis_criterion(self.dis(mel_outputs_postnet), True)
            loss_dict_discriminator['dis3'], loss_dict_discriminator['gen'] = loss_dis.data.item(), loss_gen.data.item()
            
        if not self.multigpu:
            y_hat = self.vocoder(prev,
                                self.vocoder.pad_tensor(mel_outputs_postnet, hparams.voc_pad).transpose(1, 2))
        else:
            y_hat = self.vocoder(prev,self.vocoder.module.pad_tensor(mel_outputs_postnet, hparams.voc_pad).transpose(1, 2))
        y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
        # assert (0 <= wav < 2 ** 9).all()
        loss_vocoder = self.vocoder_loss_func(y_hat, wav.unsqueeze(-1).to(device))
        self.opt_vocoder.zero_grad()

        Loss = loss_recon + loss_recon0 + loss_content + \
               self.lambda_gan * loss_gen + self.lambda_wavenet * loss_vocoder
        loss_dict['total'] = Loss.data.item()
        self.opt_encoder.zero_grad()
        self.opt_decoder.zero_grad()
        Loss.backward(retain_graph=retain_graph)
        self.opt_encoder.step()
        self.opt_decoder.step()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.vocoder.parameters(), 65504.0)
        self.opt_vocoder.step()

        if ret_content:
            return loss_recon, loss_recon0, loss_content, Loss, content
        return loss_dict, loss_dict_discriminator, loss_dict_wavenet
   

class VideoAudioGenerator(nn.Module):
    def __init__(self, dim_neck, dim_emb, dim_pre, freq, dim_spec=80, is_train=False, lr=0.001,
                 multigpu=False, 
                 lambda_wavenet=0.001, args=None,
                 residual=False, attention_map=None, use_256=False, loss_content=False,
                 test_path=None):
        super(VideoAudioGenerator, self).__init__()

        self.encoder = MyEncoder(dim_neck, freq, num_mel=dim_spec)

        self.decoder = Decoder(dim_neck, 0, dim_pre, num_mel=dim_spec)
        self.postnet = Postnet(num_mel=dim_spec)
        if use_256:
            self.video_decoder = VideoGenerator(use_256=True)
        else:
            self.video_decoder = STAGE2_G(residual=residual)
        self.use_256 = use_256
        self.lambda_wavenet = lambda_wavenet
        self.loss_content = loss_content
        self.multigpu = multigpu
        self.test_path = test_path

        self.vocoder = WaveRNN(
            rnn_dims=hparams.voc_rnn_dims,
            fc_dims=hparams.voc_fc_dims,
            bits=hparams.bits,
            pad=hparams.voc_pad,
            upsample_factors=hparams.voc_upsample_factors,
            feat_dims=hparams.num_mels,
            compute_dims=hparams.voc_compute_dims,
            res_out_dims=hparams.voc_res_out_dims,
            res_blocks=hparams.voc_res_blocks,
            hop_length=hparams.hop_size,
            sample_rate=hparams.sample_rate,
            mode=hparams.voc_mode
        )

        if is_train:
            self.criterionIdt = torch.nn.L1Loss(reduction='mean')
            self.opt_encoder = torch.optim.Adam(self.encoder.parameters(), lr=lr)
            self.opt_decoder = torch.optim.Adam(itertools.chain(self.decoder.parameters(), self.postnet.parameters()), lr=lr)
            self.opt_video_decoder = torch.optim.Adam(self.video_decoder.parameters(), lr=lr)

            self.opt_vocoder = torch.optim.Adam(self.vocoder.parameters(), lr=hparams.voc_lr)
            self.vocoder_loss_func = F.cross_entropy # Only for RAW

        if multigpu:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)
            self.video_decoder = nn.DataParallel(self.video_decoder)
            self.postnet = nn.DataParallel(self.postnet)
            self.vocoder = nn.DataParallel(self.vocoder)
    
    def optimize_parameters_video(self, dataloader, epochs, device, display_freq=10, save_freq=1000, save_dir="./",
                            experimentName="Train", initial_niter=0, load_model=None):
        writer = SummaryWriter(log_dir="logs/" + experimentName)
        if load_model is not None:
            print("Loading from %s..." % load_model)
            # self.load_state_dict(torch.load(load_model))
            d = torch.load(load_model)
            newdict = d.copy()
            for key, value in d.items():
                newkey = key
                if 'wavenet' in key:
                    newdict[key.replace('wavenet', 'vocoder')] = newdict.pop(key)
                    newkey = key.replace('wavenet', 'vocoder')
                if self.multigpu and 'module' not in key:
                    newdict[newkey.replace('.','.module.',1)] = newdict.pop(newkey)
                    newkey = newkey.replace('.', '.module.', 1)
                if newkey not in self.state_dict():
                    newdict.pop(newkey)
            print("Load " + str(len(newdict)) + " parameters!")
            self.load_state_dict(newdict, strict=False)
            print("AutoVC Model Loaded") 
        niter = initial_niter
        for epoch in range(epochs):
            self.train()
            for i, data in enumerate(dataloader):
                # print("Processing ..." + str(name))
                speaker, mel, prev, wav, video, video_large = data
                speaker, mel, prev, wav, video, video_large = speaker.to(device), mel.to(device), prev.to(device), wav.to(device), video.to(device), video_large.to(device)
                codes, code_unsample = self.encoder(mel, return_unsample=True)
                
                tmp = []
                for code in codes:
                    tmp.append(code.unsqueeze(1).expand(-1, int(mel.size(1) / len(codes)), -1))
                code_exp = torch.cat(tmp, dim=1)

                if not self.use_256:
                    v_stage1, v_stage2 = self.video_decoder(code_unsample, train=True)
                else:
                    v_stage2 = self.video_decoder(code_unsample)
                mel_outputs = self.decoder(code_exp)
                mel_outputs_postnet = self.postnet(mel_outputs.transpose(2, 1))
                mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2, 1)

                if self.loss_content:
                    _, recons_codes = self.encoder(mel_outputs_postnet, speaker, return_unsample=True)
                    loss_content = self.criterionIdt(code_unsample, recons_codes)
                else:
                    loss_content = torch.from_numpy(np.array(0))
                
                if not self.use_256:
                    loss_video = self.criterionIdt(v_stage1, video) + self.criterionIdt(v_stage2, video_large)
                else:
                    loss_video = self.criterionIdt(v_stage2, video_large)
                
                loss_recon = self.criterionIdt(mel, mel_outputs)
                loss_recon0 = self.criterionIdt(mel, mel_outputs_postnet)
                loss_vocoder = 0

                if not self.multigpu:
                    y_hat = self.vocoder(prev,
                                    self.vocoder.pad_tensor(mel_outputs_postnet, hparams.voc_pad).transpose(1, 2))
                else:
                    y_hat = self.vocoder(prev,self.vocoder.module.pad_tensor(mel_outputs_postnet, hparams.voc_pad).transpose(1, 2))
                y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
                # assert (0 <= wav < 2 ** 9).all()
                loss_vocoder = self.vocoder_loss_func(y_hat, wav.unsqueeze(-1).to(device))
                self.opt_vocoder.zero_grad()

                loss = loss_video + loss_recon + loss_recon0 + self.lambda_wavenet * loss_vocoder + loss_content

                self.opt_encoder.zero_grad()
                self.opt_decoder.zero_grad()
                self.opt_video_decoder.zero_grad()
                loss.backward()
                self.opt_encoder.step()
                self.opt_decoder.step()
                self.opt_video_decoder.step()
                self.opt_vocoder.step()



                if niter % display_freq == 0:
                    print("Epoch[%d] Iter[%d] Niter[%d] %s"
                          % (epoch, i, niter, loss.data.item()))
                    writer.add_scalars('data/Loss', {'loss':loss.data.item(),
                                                    'loss_video':loss_video.data.item(),
                                                    'loss_audio':loss_recon0.data.item()+loss_recon.data.item()}, niter)

                if niter % save_freq == 0:
                    torch.cuda.empty_cache()  # Prevent Out of Memory
                    print("Saving and Testing...", end='\t')
                    torch.save(self.state_dict(), save_dir + '/Epoch' + str(epoch).zfill(3) + '_Iter'
                               + str(niter).zfill(8) + ".pkl")
                    # self.load_state_dict(torch.load('params.pkl'))
                    self.test_audiovideo(device, writer, niter)
                    print("Done")
                    self.train()
                torch.cuda.empty_cache()  # Prevent Out of Memory
                niter += 1

    def generate(self, mel, speaker, device='cuda:0'):
        mel, speaker = mel.to(device), speaker.to(device)
        if not self.multigpu:
            codes, code_unsample = self.encoder(mel, return_unsample=True)
        else:
            codes, code_unsample = self.encoder.module(mel, speaker, return_unsample=True)
                
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1, int(mel.size(1) / len(codes)), -1))
        code_exp = torch.cat(tmp, dim=1)

        if not self.multigpu:
            if not self.use_256:
                v_stage1, v_stage2 = self.video_decoder(code_unsample, train=True)
            else:
                v_stage2 = self.video_decoder(code_unsample)
                v_stage1 = v_stage2
            mel_outputs = self.decoder(code_exp)
            mel_outputs_postnet = self.postnet(mel_outputs.transpose(2, 1))
        else:
            if not self.use_256:
                v_stage1, v_stage2 = self.video_decoder.module(code_unsample, train=True)
            else:
                v_stage2 = self.video_decoder.module(code_unsample)
                v_stage1 = v_stage2
            mel_outputs = self.decoder.module(code_exp)
            mel_outputs_postnet = self.postnet.module(mel_outputs.transpose(2, 1))
        
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2, 1)
        
        return mel_outputs_postnet, v_stage1, v_stage2
    
    def test_video(self, device):
        wav, sr = librosa.load("/mnt/lustre/dengkangle/cmu/datasets/video/obama_test.mp4", hparams.sample_rate)
        mel_basis = librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=hparams.num_mels)
        linear_spec = np.abs(
            librosa.stft(wav, n_fft=hparams.n_fft, hop_length=hparams.hop_size, win_length=hparams.win_size))
        mel_spec = mel_basis.dot(linear_spec)
        mel_db = 20 * np.log10(mel_spec)

        test_data = np.clip((mel_db + 120) / 125, 0, 1)
        test_data = torch.Tensor(pad_seq(test_data.T, hparams.freq)).unsqueeze(0).to(device)
        with torch.no_grad():
            codes, code_exp = self.encoder.module(test_data, return_unsample=True)
            v_mid, v_hat = self.video_decoder.module(code_exp, train=True)

        reader = imageio.get_reader("/mnt/lustre/dengkangle/cmu/datasets/video/obama_test.mp4", 'ffmpeg', fps=20)
        frames = []
        for i, im in enumerate(reader):
            frames.append(np.array(im).transpose(2, 0, 1))
        frames = (np.array(frames) / 255 - 0.5) / 0.5
        return frames, v_mid[0:1], v_hat[0:1]

    def test_audiovideo(self, device, writer, niter):
        source_path = self.test_path

        mel_basis80 = librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=80)

        wav, sr = librosa.load(source_path, hparams.sample_rate)
        wav = preemphasis(wav, hparams.preemphasis, hparams.preemphasize)

        linear_spec = np.abs(
            librosa.stft(wav, n_fft=hparams.n_fft, hop_length=hparams.hop_size, win_length=hparams.win_size))
        mel_spec = mel_basis80.dot(linear_spec)
        mel_db = 20 * np.log10(mel_spec)
        source_spec = np.clip((mel_db + 120) / 125, 0, 1)
        
        source_embed = torch.from_numpy(np.array([0, 1])).float().unsqueeze(0)
        source_wav = wav

        source_spec = torch.Tensor(pad_seq(source_spec.T, hparams.freq)).unsqueeze(0)
        # print(source_spec.shape)
        
        with torch.no_grad():
            generated_spec, v_mid, v_hat = self.generate(source_spec, source_embed ,device)

        generated_spec, v_mid, v_hat = generated_spec.cpu(), v_mid.cpu(), v_hat.cpu()

        print("Generating Wavfile...")
        with torch.no_grad():
            if not self.multigpu:
                generated_wav = inv_preemphasis(self.vocoder.generate(generated_spec.to(device).transpose(2, 1), False, None, None, mu_law=True), hparams.preemphasis, hparams.preemphasize)
            
            else:
                generated_wav = inv_preemphasis(self.vocoder.module.generate(generated_spec.to(device).transpose(2, 1), False, None, None, mu_law=True), hparams.preemphasis, hparams.preemphasize)


        writer.add_video('generated', (v_hat.numpy()+1)/2, global_step=niter)
        writer.add_video('mid', (v_mid.numpy()+1)/2, global_step=niter)
        writer.add_audio('ground_truth', source_wav, niter, sample_rate=hparams.sample_rate)
        writer.add_audio('generated_wav', generated_wav, niter, sample_rate=hparams.sample_rate)
