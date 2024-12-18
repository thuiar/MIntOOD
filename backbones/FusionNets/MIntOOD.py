import torch
import numpy as np
import torch.nn.functional as F
import math
from losses import loss_map
from ..SubNets.FeatureNets import BERTEncoder, SubNet
from ..SubNets.transformers_encoder.transformer import TransformerEncoder
from .sampler import MIntOODSampler
from torch import nn
from ..SubNets import text_backbones_map
from data.__init__ import benchmarks
from ..SubNets.AlignNets import AlignSubNet
from torch.nn.parameter import Parameter
from .MIntOOD_Fusion import Fusion

activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}
__all__ = ['MIntOOD']

class CosNorm_Classifier(nn.Module):

    def __init__(self, in_dims, out_dims, scale = 32, device = None):

        super(CosNorm_Classifier, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.weight = Parameter(torch.Tensor(out_dims, in_dims).to(device))
        self.scale = scale
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, binary_scores = None,  *args):

        norm_x = torch.norm(input, 2, 1, keepdim=True)
        ex = input / norm_x
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)

        return torch.mm(ex * self.scale, ew.t())

class MLP_head(nn.Module):
    
    def __init__(self, args, num_classes):
        
        super(MLP_head, self).__init__()
        self.args = args
        self.num_classes = num_classes

        if num_classes == 2:

            self.layer1 = nn.Linear(args.base_dim, args.mlp_hidden_size)
            self.layer2 = nn.Linear(args.mlp_hidden_size, args.mlp_hidden_size)   
     
            self.relu = nn.ReLU()
            self.gelu = nn.GELU()
            self.dropout = nn.Dropout(p=args.mlp_dropout)
            self.output_layer = nn.Linear(args.mlp_hidden_size, num_classes)

        else:

            self.relu = nn.ReLU()
            self.gelu = nn.GELU()
            if args.ablation_type == 'wo_cosine':
                self.output_layer_1 = nn.Linear(args.base_dim, args.num_labels)
            else:
                self.output_layer_1 = CosNorm_Classifier(args.base_dim, args.num_labels, args.scale, args.device)
            
            if args.ablation_type != 'wo_contrast':
                self.contrast_head = nn.Sequential(
                    nn.Linear(args.base_dim, args.num_labels)
                )

    def adjust_scores(self, scores):
        eps = 1e-6
        adjusted_scores = scores / (1 - scores + eps)
        return adjusted_scores

    def forward(self, x, binary_scores = None, mode = 'ind', return_mlp=False):

        if self.num_classes == 2:

            x = self.relu(self.layer1(x))
            x = self.dropout(x)
            x = self.relu(self.layer2(x))
            x = self.dropout(x)
            x = self.output_layer(x)
            
            return x

        else:
            if binary_scores is not None:
                
                binary_scores = self.adjust_scores(binary_scores)
                binary_scores = binary_scores.unsqueeze(1).expand(-1, x.shape[1])
                fusion_x = x * binary_scores
               
                logits = self.output_layer_1(fusion_x)

                if self.args.ablation_type != 'wo_contrast':
                    contrast_logits = self.contrast_head(x)
                else:
                    contrast_logits = logits
                
                
                if return_mlp:
                    return fusion_x, logits, contrast_logits
                else:
                    return logits, contrast_logits
            
            else:
                fusion_x = x
                logits = self.output_layer_1(fusion_x)
                contrast_logits = self.contrast_head(x)

                if return_mlp:
                    return fusion_x, logits, contrast_logits
                else:
                    return logits, contrast_logits

    def vim(self):
        w = self.output_layer_1.weight
        b = torch.zeros(w.size(0))
       
        return w, b

class MMEncoder(nn.Module):

    def __init__(self, args):

        super(MMEncoder, self).__init__()
        self.model = Fusion(args)
        self.sampler = MIntOODSampler(args)

    def forward(self, text_feats, video_feats, audio_feats, labels = None, ood_sampling = False, data_aug = False, probs = None, binary = False, ood_elems = None):
        
        if ood_sampling:
            pooled_output, mixed_labels = self.model(text_feats, video_feats, audio_feats, self.sampler, labels, ood_sampling = ood_sampling, data_aug = data_aug, probs = probs, binary = binary, ood_elems = ood_elems)
            return pooled_output, mixed_labels
        else:
            pooled_output = self.model(text_feats, video_feats, audio_feats, self.sampler, labels, ood_sampling = ood_sampling, data_aug = data_aug, probs = probs, binary = binary, ood_elems = ood_elems)
            return pooled_output
        
 
