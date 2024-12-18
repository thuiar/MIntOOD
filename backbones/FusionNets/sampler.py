import torch
import numpy as np
import random
from torch import nn
from scipy.spatial.distance import cdist

def mixup_data(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam

class MIntOODSampler(nn.Module):

    def __init__(self, args):
        super(MIntOODSampler, self).__init__()
        self.ood_label_id = args.ood_label_id
        self.args = args

    def alternate_mixup(self, data1, data2):
        mixed_data = torch.zeros_like(data1)
        for i in range(data1.size(0)):
            if i % 2 == 0:
                mixed_data[i, :] = data1[i, :]
            else:
                mixed_data[i, :] = data2[i, :]
        return mixed_data

    def forward(self, ind_text_feats, ind_video_data, ind_audio_data, ind_label_ids, extended_attention_mask, attention_mask, device=None, binary = False, ood_elems = None):

        if binary:
            num_ood = int(len(ind_text_feats) * self.args.binary_multiple_ood)
        else:    
            num_ood = int(len(ind_text_feats) * self.args.multiple_ood)

        ood_text_list, ood_video_list, ood_audio_list, ood_mask_list, ood_attention_mask_list = [], [], [], [], []
        text_seq_length, video_seq_length, audio_seq_length = ind_text_feats.shape[1], ind_video_data.shape[1], ind_audio_data.shape[1]

        select_elems = []

        if self.args.ablation_type == 'sampler_beta':

            while len(ood_text_list) < num_ood:
                
                cdt = np.random.choice(ind_label_ids.size(0), 2, replace=False)
                
                if len(set(ind_label_ids[cdt].tolist())) >= 2:
                    s = mixup_data(self.args.alpha)
                    
                    ood_text = (s * ind_text_feats[cdt[0]] + (1 - s) * ind_text_feats[cdt[1]])
                    ood_video = (s * ind_video_data[cdt[0]] + (1 - s) * ind_video_data[cdt[1]])
                    ood_audio = (s * ind_audio_data[cdt[0]] + (1 - s) * ind_audio_data[cdt[1]])

                    
                    lengths = [torch.sum(extended_attention_mask[cdt[i]]).item() for i in range(2)]
                    idx = cdt[np.argmin(lengths)]
                    ood_mask = extended_attention_mask[idx]
                    ood_attention_mask = attention_mask[idx]

                    ood_text_list.append(ood_text)
                    ood_video_list.append(ood_video)
                    ood_audio_list.append(ood_audio)
                    ood_mask_list.append(ood_mask)
                    ood_attention_mask_list.append(ood_attention_mask)

                    
                    select_elems.append([cdt[0], s, cdt[1]])

        else:
            while len(ood_text_list) < num_ood:
                
                if self.args.select_number_min == self.args.select_number_max:
                    select_number = self.args.select_number_min
                else:    
                    select_number = np.random.randint(self.args.select_number_min, self.args.select_number_max + 1)

                if ind_label_ids.size(0) >= select_number:
                    cdt = np.random.choice(ind_label_ids.size(0), select_number, replace=False)
                    
                    if len(set(ind_label_ids[cdt].tolist())) >= 2:
                        s = np.random.dirichlet(alpha=[self.args.alpha] * select_number)
                        
                        ood_text = sum(s[i] * ind_text_feats[cdt[i]] for i in range(select_number))
                        ood_video = sum(s[i] * ind_video_data[cdt[i]] for i in range(select_number))
                        ood_audio = sum(s[i] * ind_audio_data[cdt[i]] for i in range(select_number))

                        
                        lengths = [torch.sum(extended_attention_mask[cdt[i]]).item() for i in range(select_number)]
                        idx = cdt[np.argmin(lengths)]
                        ood_mask = extended_attention_mask[idx]
                        ood_attention_mask = attention_mask[idx]

                        ood_text_list.append(ood_text)
                        ood_video_list.append(ood_video)
                        ood_audio_list.append(ood_audio)
                        ood_mask_list.append(ood_mask)
                        ood_attention_mask_list.append(ood_attention_mask)

                        
                        select_elems.append([cdt.tolist(), s.tolist()])

        if ind_text_feats.ndim == 3:
            ood_text_feats = torch.cat(ood_text_list, dim = 0).view(num_ood, text_seq_length, -1)
            ood_mask_feats = torch.cat(ood_mask_list, dim = 0).view(num_ood, extended_attention_mask.shape[1], extended_attention_mask.shape[2], extended_attention_mask.shape[3])
            ood_attention_mask_feats = torch.cat(ood_attention_mask_list, dim = 0).view(num_ood, -1)
        elif ind_text_feats.ndim == 2:
            ood_text_feats = torch.cat(ood_text_list, dim = 0).view(num_ood, -1)
            
        if ind_video_data.ndim == 3:
            ood_video_feats = torch.cat(ood_video_list, dim = 0).view(num_ood, video_seq_length, -1)
        elif ind_video_data.ndim == 2:
            ood_video_feats = torch.cat(ood_video_list, dim = 0).view(num_ood, -1)
        
        if ind_audio_data.ndim == 3:
            ood_audio_feats = torch.cat(ood_audio_list, dim = 0).view(num_ood, audio_seq_length, -1)
        elif ind_audio_data.ndim == 2:
            ood_audio_feats = torch.cat(ood_audio_list, dim = 0).view(num_ood, -1)
            
        mix_text = torch.cat((ind_text_feats, ood_text_feats), dim = 0)

        mix_video = torch.cat((ind_video_data, ood_video_feats), dim = 0)
        mix_audio = torch.cat((ind_audio_data, ood_audio_feats), dim = 0)
        mix_mask = torch.cat((extended_attention_mask, ood_mask_feats), dim = 0)
        mix_attention_mask = torch.cat((attention_mask, ood_attention_mask_feats), dim = 0)

        semi_label_ids = torch.cat((ind_label_ids.cpu(), torch.tensor([self.ood_label_id] * num_ood)), dim=0)
        binary_label_ids = torch.cat((torch.tensor([1] * len(ind_text_feats)) , torch.tensor([0] * num_ood)), dim=0)

        mix_data = {}
        mix_data['text'] = mix_text.to(device)
        mix_data['video'] = mix_video.to(device)
        mix_data['audio'] = mix_audio.to(device)
        mix_data['mask'] = mix_mask.to(device)
        mix_data['attention_mask'] = mix_attention_mask.to(device)

        mix_labels = {
            'ind': ind_label_ids.to(device),
            'semi': semi_label_ids.to(device),
            'binary': binary_label_ids.to(device),
            'select_elems': select_elems
        }
    
        return mix_data, mix_labels 
