import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm as wn
from utils import *
import pdb

class nin(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(nin, self).__init__()
        self.lin_a = wn(nn.Linear(dim_in, dim_out))
        self.dim_out = dim_out

    def forward(self, x):
        og_x = x
        # assumes pytorch ordering
        """ a network in network layer (1x1 CONV) """
        # TODO : try with original ordering
        x = x.permute(0, 2, 3, 1)
        shp = [int(y) for y in x.size()]
        out = self.lin_a(x.contiguous().view(shp[0]*shp[1]*shp[2], shp[3]))
        shp[-1] = self.dim_out
        out = out.view(shp)
        return out.permute(0, 3, 1, 2)


class down_shifted_conv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,3), stride=(1,1),
                    shift_output_down=False, norm='weight_norm'):
        super(down_shifted_conv2d, self).__init__()

        assert norm in [None, 'batch_norm', 'weight_norm']
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride)
        self.shift_output_down = shift_output_down
        self.norm = norm
        self.pad  = nn.ZeroPad2d((int((filter_size[1] - 1) / 2), # pad left
                                  int((filter_size[1] - 1) / 2), # pad right
                                  filter_size[0] - 1,            # pad top
                                  0) )                           # pad down

        if norm == 'weight_norm':
            self.conv = wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_down :
            self.down_shift = lambda x : down_shift(x, pad=nn.ZeroPad2d((0, 0, 1, 0)))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.down_shift(x) if self.shift_output_down else x


class down_shifted_deconv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,3), stride=(1,1)):
        super(down_shifted_deconv2d, self).__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size, stride,
                                            output_padding=1))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        return x[:, :, :(xs[2] - self.filter_size[0] + 1),
                 int((self.filter_size[1] - 1) / 2):(xs[3] - int((self.filter_size[1] - 1) / 2))]


class down_right_shifted_conv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,2), stride=(1,1),
                    shift_output_right=False, norm='weight_norm'):
        super(down_right_shifted_conv2d, self).__init__()

        assert norm in [None, 'batch_norm', 'weight_norm']
        self.pad = nn.ZeroPad2d((filter_size[1] - 1, 0, filter_size[0] - 1, 0))
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride=stride)
        self.shift_output_right = shift_output_right
        self.norm = norm

        if norm == 'weight_norm':
            self.conv = wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_right :
            self.right_shift = lambda x : right_shift(x, pad=nn.ZeroPad2d((1, 0, 0, 0)))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.right_shift(x) if self.shift_output_right else x


class down_right_shifted_deconv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,2), stride=(1,1),
                    shift_output_right=False):
        super(down_right_shifted_deconv2d, self).__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size,
                                                stride, output_padding=1))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        x = x[:, :, :(xs[2] - self.filter_size[0] + 1):, :(xs[3] - self.filter_size[1] + 1)]
        return x

class FiLM(nn.Module):
    def __init__(self, embedding_dim, num_filters, gamma_scale=1.5, hidden_dim=32):
        super(FiLM, self).__init__()
        
        # Projection network to generate gamma and beta parameters
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_filters),  # 4× because we need 2× for gamma and 2× for beta
            nn.ReLU(),
        )

        self.gamma_scale = nn.Parameter(torch.tensor(gamma_scale))
        # Initialize weights properly to avoid gradient issues
        nn.init.xavier_uniform_(self.projection[0].weight, gain=1.0)
        nn.init.zeros_(self.projection[0].bias)
        nn.init.xavier_uniform_(self.projection[3].weight, gain=1.0)
        nn.init.zeros_(self.projection[3].bias)
        
    def forward(self, x, class_embedding):
        og_x = x
        # Generate modulation parameters
        gamma_beta = self.projection(class_embedding)
        
        # Split into gamma and beta parameters
        gamma, beta = gamma_beta.chunk(2, dim=1)
        
        # Reshape for broadcasting over spatial dimensions
        gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1)
        beta = beta.view(beta.size(0), beta.size(1), 1, 1)
        
        # Apply the FiLM transformation
        output = og_x + (self.gamma_scale * gamma) * x + beta
        
        return output

'''
skip connection parameter : 0 = no skip connection
                            1 = skip connection where skip input size === input size
                            2 = skip connection where skip input size === 2 * input size
'''
class gated_resnet(nn.Module):
    def __init__(self, num_filters, conv_op, nonlinearity=concat_elu, skip_connection=0, embedding_dim=None):
        super(gated_resnet, self).__init__()
        self.skip_connection = skip_connection
        self.nonlinearity = nonlinearity
        self.conv_input = conv_op(2 * num_filters, num_filters) # cuz of concat elu

        if skip_connection != 0 :
            self.nin_skip = nin(2 * skip_connection * num_filters, num_filters)

        # to project from filter + embedding dim back to original filter dim
        # nin performs a 1x1 conv to convert from one dim to another
        if embedding_dim is not None:
            self.film = FiLM(embedding_dim, 4 * num_filters, gamma_scale=2.0, hidden_dim=32)

        # apply pre-norm, similar as from the transformer layer in PA2
        # use group norm instead because we're dealing with conv layers that have varying output dims
        self.gn1 = nn.GroupNorm(1, num_filters)
        self.gn2 = nn.GroupNorm(1, 2 * num_filters)

        self.dropout = nn.Dropout2d(0.5)
        self.conv_out = conv_op(2 * num_filters, 2 * num_filters)


    def forward(self, og_x, a=None, class_embedding=None):
        x = self.conv_input(self.nonlinearity(og_x))
        if a is not None :
            x += self.nin_skip(self.nonlinearity(a))
        
        x = self.gn1(x)
        x = self.nonlinearity(x)
        # pdb.set_trace()
        
        x = self.dropout(x)

        x = self.conv_out(x)
        x = self.gn2(x)
        # pdb.set_trace()

        if class_embedding is not None:
            x = self.film(x, class_embedding)

        a, b = torch.chunk(x, 2, dim=1)
        c3 = a * F.sigmoid(b)
        return og_x + c3