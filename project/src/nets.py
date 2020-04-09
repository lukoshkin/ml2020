import numpy as np
import torch 
import torch.nn as nn

from causal_convolution import DCConvStack

class RNNSeqEncoder(nn.Module):
    def __init__(self, emb_size, code_size):
        super().__init__()
        self.gru1 = nn.GRU(emb_size, code_size)
        self.gru2 = nn.GRU(code_size, code_size)
        self.proj = nn.Sequential(
            nn.Linear(code_size, code_size),
            nn.LeakyReLU(.2))
        
    def forward(self, X):
        H,res_h = self.gru1(X)
        H,last= self.gru2(H, self.proj(res_h))
        code = last + res_h
        return code
        
class RNNSeqDecoder(nn.Module):
    def __init__(self, emb_size, code_size, seq_length):
        super().__init__()
        self.gru1 = nn.GRU(code_size, code_size)
        self.gru2 = nn.GRU(code_size, emb_size)
        self.proj1 = nn.Sequential(
            nn.Linear(code_size, code_size),
            nn.LeakyReLU(.2, inplace=True))
        self.proj2 = nn.Sequential(
            nn.Linear(code_size, emb_size),
            nn.LeakyReLU(.2, inplace=True))
        self.proj3 = nn.Sequential(
            nn.Linear(code_size, emb_size))
        self._shape = (seq_length, -1, code_size)
        
    def forward(self, code):
        H = code.expand(self._shape)
        H,last = self.gru1(H, self.proj1(code))
        X,_ = self.gru2(H, self.proj2(last))
        return X + self.proj3(H)

class RNNSeqAE(nn.Module):
    def __init__(self, emb_size, code_size, seq_length):
        super().__init__()
        self.enc = RNNSeqEncoder(emb_size, code_size)
        self.dec = RNNSeqDecoder(emb_size, code_size, seq_length)
    
    def forward(self, X):
        return self.dec(self.enc(X))



# CRAZY IMPLEMENTATION (CLOSE TO MEMORIZATION ?)
class CrazyRNNSeqEncoder(nn.Module):
    def __init__(self, emb_size, code_size):
        super().__init__()
        self.gru1 = nn.GRU(emb_size, code_size)
        self.gru2 = nn.GRU(code_size, code_size)
        self.proj = nn.Sequential(
            nn.Linear(code_size, code_size),
            nn.LeakyReLU(.2))
        
    def forward(self, X):
        H,res_h = self.gru1(X)
        H,last= self.gru2(H, self.proj(res_h))
        code = last + res_h
        return code
        

class CrazyRNNSeqDecoder(nn.Module):
    def __init__(self, emb_size, code_size, seq_length):
        super().__init__()
        self.gru1 = nn.GRU(code_size, code_size)
        self.gru2 = nn.GRU(code_size, emb_size)
        self.proj1 = nn.Sequential(
            nn.Linear(code_size, code_size),
            nn.LeakyReLU(.2, inplace=True))
        self.proj2 = nn.Sequential(
            nn.Linear(code_size, emb_size),
            nn.LeakyReLU(.2, inplace=True))
        self.proj3 = nn.Sequential(
            nn.Linear(seq_length*code_size, seq_length*emb_size)
        )
        self._shape = (seq_length, -1, code_size)
        self.new_size = code_size*seq_length
        self.new_shape = (-1, seq_length, emb_size)
        
    def forward(self, code):
        H = code.expand(self._shape)
        H,last = self.gru1(H, self.proj1(code))
        X,_ = self.gru2(H, self.proj2(last))
        H = H.permute(1,0,2).reshape(-1, self.new_size)
        H = self.proj3(H).reshape(self.new_shape)
        return X + H.permute(1,0,2)


class CrazyRNNSeqAE(nn.Module):
    def __init__(self, emb_size, code_size, seq_length):
        super().__init__()
        self.enc = CrazyRNNSeqEncoder(emb_size, code_size)
        self.dec = CrazyRNNSeqDecoder(emb_size, code_size, seq_length)
    
    def forward(self, X):
        return self.dec(self.enc(X))



class ConvEncoder(nn.Module):
    def __init__(self, emb_size, code_size, channels, kernel_size=32):
        super().__init__()
        channels = channels.copy()
        channels.insert(0, emb_size)

        self.tb = DCConvStack(channels, kernel_size, len(channels)-1)
        self.fc = nn.Linear(channels[-1], code_size)

    def forward(self, X):
        H,_ = self.tb(X).max(axis=-1)
        return self.fc(H)


class ConvDecoder(nn.Module):
    def __init__(
        self, emb_size, code_size,
        seq_length, channels, kernel_size=32):
        super().__init__()
        channels = channels.copy()
        channels.append(emb_size)

        self.tb = DCConvStack(channels, kernel_size, len(channels)-1)
        self.fc = nn.Linear(code_size, channels[0])
        self.leaky = nn.LeakyReLU(.2, inplace=True)
        self._shape = (channels[0], seq_length)

    def forward(self, code):
        H = self.leaky(self.fc(code))
        H = H[...,None].expand(-1, *self._shape)
        return self.tb(H)#.flip(-1)


class ConvSeqAE(nn.Module):
    def __init__(
        self, emb_size, code_size,
        seq_length, channels, kernel_size=32):
        super().__init__()

        self.enc = ConvEncoder(
            emb_size, code_size, channels, kernel_size)
        self.dec = ConvDecoder(
            emb_size, code_size,
            seq_length, channels[::-1], kernel_size)

    def forward(self, X):
        return self.dec(self.enc(X))


class SeqClassifier(ConvEncoder):
    def __init__(self, emb_size, n_classes, channels, kernel_size=7):
        super().__init__(emb_size, n_classes, channels, kernel_size)

    def forward(self, X):
        return super().forward(X)

    def predict_probs(self, X):
        return super().forward(X).softmax(1)

    def predict(self, X):
        return super().forward(X).argmax(1)



class LandscapeInverse(nn.Module):
# Input: (batch_size, feature_level, num_layers, n_bins)
# The last three form 'shape' 3-tuple
    def __init__(
        self, emb_size, shape,
        seq_length, channels, kernel_size=28):
        super().__init__()

        channels.append(emb_size)
        channels.insert(0, shape[0]*shape[1])
        self.tb = DCConvStack(channels, kernel_size, len(channels)-1)
        self.proj = nn.Sequential(
                    nn.Linear(shape[2], seq_length),
                    nn.BatchNorm1d(shape[0]*shape[1]),
                    nn.LeakyReLU(.2, inplace=True))

    def forward(self, landscape):
        return self.tb(self.proj(landscape.flatten(1,2)))

    # if landscape is not a torch.Tensor, substitute `.flatten(1,2)`
    # with `torch.flatten(<>, 1, 2)`



# forward's input:
#    landscape (batch_size, feature_level, num_layers, n_bins)
#    The last 3 form 'shape' 3-tuple

#    code (batch_size, code_size)

class StackDecoder(nn.Module):
    def __init__(
        self, emb_size, code_size, shape,
        seq_length, channels, kernel_size=28):
        super().__init__()
        channels = channels.copy()
        channels.append(emb_size)

        self._shape = (channels[0], seq_length)
        self.tb = DCConvStack(channels, kernel_size, len(channels)-1)
        self.proj1 = nn.Sequential(
                    nn.Linear(np.prod(shape), code_size),
                    nn.BatchNorm1d(code_size),
                    nn.LeakyReLU(.2, inplace=True))
        self.proj2 = nn.Sequential(
                    nn.Linear(code_size*2, channels[0]),
                    nn.BatchNorm1d(channels[0]),
                    nn.LeakyReLU(.2, inplace=True))

    def forward(self, code, landscape):
        H = torch.cat([code, self.proj1(landscape.flatten(1))], 1)
        H = self.proj2(H)[...,None].expand(-1, *self._shape)
        return self.tb(H)

    # if landscapy is not a torch.Tensor, substitute `.flatten(1,2)`
    # with `torch.flatten(<>, 1, 2)`

class TDASAE(nn.Module):
    def __init__(
        self, emb_size, code_size, seq_length,
        shape, channels, kernel_size=28):
        super().__init__()

        self.enc = ConvEncoder(
            emb_size, code_size, channels, kernel_size)
        self.dec = StackDecoder(
            emb_size, code_size, shape,
            seq_length, channels[::-1], kernel_size)

    def forward(self, X, landscape):
        code = self.enc(X)
        return self.dec(self.enc(X), landscape)
