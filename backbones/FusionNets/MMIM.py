"""
From: https://github.com/declare-lab/Multimodal-Infomax
Paper: Improving Multimodal Fusion with Hierarchical Mutual Information Maximization for Multimodal Sentiment Analysis
"""

import torch
from ..SubNets import text_backbones_map
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.parameter import Parameter
import math



__all__ = ['MMIM']


class RNNEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super().__init__()
        self.bidirectional = bidirectional

        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=False)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear((2 if bidirectional else 1) * hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''

        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(packed_sequence)
        
        if self.bidirectional:
            h = self.dropout(torch.cat((final_states[0][0],final_states[0][1]),dim=-1))
        else:
            h = self.dropout(final_states[0].squeeze())
            
        y_1 = self.linear_1(h)
        return y_1

class MMILB(nn.Module):
    """Compute the Modality Mutual Information Lower Bound (MMILB) given bimodal representations.
    Args:
        x_size (int): embedding size of input modality representation x
        y_size (int): embedding size of input modality representation y
        mid_activation(int): the activation function in the middle layer of MLP
        last_activation(int): the activation function in the last layer of MLP that outputs logvar
    """
    def __init__(self, x_size, y_size, mid_activation='ReLU', last_activation='Tanh'):
        super(MMILB, self).__init__()
        try:
            self.mid_activation = getattr(nn, mid_activation)
            self.last_activation = getattr(nn, last_activation)
        except:
            raise ValueError("Error: CLUB activation function not found in torch library")
        self.mlp_mu = nn.Sequential(
            nn.Linear(x_size, y_size),
            self.mid_activation(),
            nn.Linear(y_size, y_size)
        )
        self.mlp_logvar = nn.Sequential(
            nn.Linear(x_size, y_size),
            self.mid_activation(),
            nn.Linear(y_size, y_size),
        )
        self.entropy_prj = nn.Sequential(
            nn.Linear(y_size, y_size // 4),
            nn.Tanh()
        )

    def forward(self, x, y):
        """ Forward lld (gaussian prior) and entropy estimation, partially refers the implementation
        of https://github.com/Linear95/CLUB/blob/master/MI_DA/MNISTModel_DANN.py
            Args:
                x (Tensor): x in above equation, shape (bs, x_size)
                y (Tensor): y in above equation, shape (bs, y_size)
        """

        mu, logvar = self.mlp_mu(x), self.mlp_logvar(x) # (bs, hidden_size)

        positive = -(mu - y) ** 2 / 2. / torch.exp(logvar)
        lld = torch.mean(torch.sum(positive, -1))

        return lld

class CPC(nn.Module):
    """
        Contrastive Predictive Coding: score computation. See https://arxiv.org/pdf/1807.03748.pdf.

        Args:
            x_size (int): embedding size of input modality representation x
            y_size (int): embedding size of input modality representation y
    """
    def __init__(self, x_size, y_size, n_layers=1, activation='Tanh'):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.layers = n_layers
        self.activation = getattr(nn, activation)
        if n_layers == 1:
            self.net = nn.Linear(
                in_features=y_size,
                out_features=x_size
            )
        else:
            net = []
            for i in range(n_layers):
                if i == 0:
                    net.append(nn.Linear(self.y_size, self.x_size))
                    net.append(self.activation())
                else:
                    net.append(nn.Linear(self.x_size, self.x_size))
            self.net = nn.Sequential(*net)
        
    def forward(self, x, y):
        """Calulate the score 
        """
        # import ipdb;ipdb.set_trace()
        x_pred = self.net(y)    # bs, emb_size

        # normalize to unit sphere
        x_pred = x_pred / x_pred.norm(dim=1, keepdim=True)
        x = x / x.norm(dim=1, keepdim=True)

        pos = torch.sum(x*x_pred, dim=-1)   # bs
        neg = torch.logsumexp(torch.matmul(x, x_pred.t()), dim=-1)   # bs
        nce = -(pos - neg).mean()
        return nce

class Fusion(nn.Module): #SubNet  # fusion network consisting of stacks of linear activation layers
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, args, in_size, hidden_size, n_class, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(Fusion, self).__init__() #SubNet
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, n_class)


    def forward(self, x, binary_inputs = None):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        # normed = self.norm(x)
        dropped = self.drop(x)
        y_1 = torch.tanh(self.linear_1(dropped))
        y_2 = torch.tanh(self.linear_2(y_1))
        y_3 = self.linear_3(y_2)

        
        return y_2, y_3

class MMIM(nn.Module):
    
    def __init__(self, args):
        """Construct MultiMoldal InfoMax model.
        Args: 
            args (dict): a dict stores training and model argsurations
        """
        # Base Encoders
        super(MMIM,self).__init__()
        

        output_dim = args.num_labels
        self.args = args
        self.add_va = args.add_va
        args.d_tout = args.text_feat_dim
        
        text_backbone = text_backbones_map[args.text_backbone]
        self.text_subnet = text_backbone(args)

        self.visual_enc = RNNEncoder(    # visual
            in_size = args.video_feat_dim,
            hidden_size = args.d_vh,
            out_size = args.d_vout,
            num_layers = args.n_layer,
            dropout = args.dropout_v if args.n_layer > 1 else 0.0,
            bidirectional = args.bidirectional
        )
        self.acoustic_enc = RNNEncoder(   # acoustic
            in_size = args.audio_feat_dim,
            hidden_size = args.d_ah,
            out_size = args.d_aout,
            num_layers = args.n_layer,
            dropout = args.dropout_a if args.n_layer > 1 else 0.0,
            bidirectional = args.bidirectional
        )

        # For MI maximization, calculating lower bound for each modality
        self.mi_tv = MMILB(
            x_size = args.d_tout,
            y_size = args.d_vout,
            mid_activation = args.mmilb_mid_activation,
            last_activation = args.mmilb_last_activation
        )

        self.mi_ta = MMILB(
            x_size = args.d_tout,
            y_size = args.d_aout,
            mid_activation = args.mmilb_mid_activation,
            last_activation = args.mmilb_last_activation
        )

        if args.add_va:
            self.mi_va = MMILB(
                x_size = args.d_vout,
                y_size = args.d_aout,
                mid_activation = args.mmilb_mid_activation,
                last_activation = args.mmilb_last_activation
            )

        dim_sum = args.d_aout + args.d_vout + args.d_tout

        # CPC MI bound, contractive predictive coding for each modality
        self.cpc_zt = CPC(
            x_size = args.d_tout, # to be predicted
            y_size = args.d_prjh,
            n_layers = args.cpc_layers,
            activation = args.cpc_activation
        )
        self.cpc_zv = CPC(
            x_size = args.d_vout,
            y_size = args.d_prjh,
            n_layers = args.cpc_layers,
            activation = args.cpc_activation
        )
        self.cpc_za = CPC(
            x_size = args.d_aout,
            y_size = args.d_prjh,
            n_layers = args.cpc_layers,
            activation = args.cpc_activation
        )

        # Trimodal Settings
        self.fusion_prj = Fusion(
            args,
            in_size = dim_sum,
            hidden_size = args.d_prjh,
            n_class = output_dim,
            dropout = args.dropout_prj
        )
        
        args.feat_size = args.d_prjh
  
    def forward(self, text_feats, video_data, audio_data, mode = None, binary_inputs = None, feature_ext = False):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        For Bert input, the length of text is "seq_len + 2"
        """
        video_feats = video_data['feats'].float()
        audio_feats = audio_data['feats'].float()
        video_lengths = video_data['lengths'].int().detach().cpu()
        audio_lengths = audio_data['lengths'].int().detach().cpu()

        enc_word = self.text_subnet(text_feats) # (batch_size, seq_len, emb_size)
        
        if feature_ext:
            return enc_word, video_feats, audio_feats
        
        text_h = enc_word[:,0,:] # (batch_size, emb_size)

        audio_h = self.acoustic_enc(audio_feats, audio_lengths)
        vision_h = self.visual_enc(video_feats, video_lengths)

        lld_tv = self.mi_tv(x=text_h, y=vision_h)
        
        lld_ta = self.mi_ta(x=text_h, y=audio_h)

        if self.add_va:
            lld_va = self.mi_va(x=vision_h, y=audio_h)

        fusion, preds = self.fusion_prj(torch.cat([text_h, audio_h, vision_h], dim=1), binary_inputs = binary_inputs)
   
        nce_t = self.cpc_zt(text_h, fusion)
        nce_v = self.cpc_zv(vision_h, fusion)
        nce_a = self.cpc_za(audio_h, fusion)
        
        nce = nce_t + nce_v + nce_a   # cpc loss

        lld = lld_tv + lld_ta + (lld_va if self.add_va else 0.0)

        if mode == 'train':
            res = {
                'Feature_t': text_h,
                'Feature_a': audio_h,
                'Feature_v': vision_h, 
                'Feature_f': fusion,
                'lld': lld,  # MAE loss
                'nce': nce,   # cpc loss
                'M': preds
            }

            return res
        else:
            return preds, fusion
    
    def vim(self):

        return self.fusion_prj.linear_3.weight, self.fusion_prj.linear_3.bias


