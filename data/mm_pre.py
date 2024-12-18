from torch.utils.data import Dataset
import torch
import numpy as np

__all__ = ['MMDataset']

class MMDataset(Dataset):
        
    def __init__(self, label_ids, text_data, video_data, audio_data):
        
        self.label_ids = label_ids
        self.text_data = text_data
        self.video_data = video_data
        self.audio_data = audio_data
        self.size = len(self.text_data)
        # print('111111111111', len(self.text_data))
        # print('22222222222', len(self.video_data['feats']))
        # print('3333333333', len(self.audio_data['feats']))

        
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # print('1111111', index)
        # print('222222', self.label_ids)
        sample = {
            'label_ids': torch.tensor(self.label_ids[index]), 
            'text_feats': torch.tensor(self.text_data[index]),
            'video_feats': torch.tensor(np.array(self.video_data['feats'][index])),
            'video_lengths': torch.tensor(np.array(self.video_data['lengths'][index])),
            'audio_feats': torch.tensor(np.array(self.audio_data['feats'][index])),
            'audio_lengths': torch.tensor(np.array(self.audio_data['lengths'][index]))
        } 
        return sample


class TCL_MAPDataset(Dataset):
        
    def __init__(self, label_ids, text_feats, video_feats, audio_feats, cons_text_feats, condition_idx):
        
        
        self.label_ids = label_ids
        self.text_feats = text_feats
        self.cons_text_feats = cons_text_feats
        self.condition_idx = condition_idx
        self.video_feats = video_feats
        self.audio_feats = audio_feats
        self.size = len(self.text_feats)

    def __len__(self):
        return self.size

    def __getitem__(self, index):

        sample = {
            'label_ids': torch.tensor(self.label_ids[index]), 
            'text_feats': torch.tensor(self.text_feats[index]),
            'video_feats': torch.tensor(self.video_feats['feats'][index]),
            'audio_feats': torch.tensor(self.audio_feats['feats'][index]),
            'cons_text_feats': torch.tensor(self.cons_text_feats[index]),
            'condition_idx': torch.tensor(self.condition_idx[index])
        } 
        return sample