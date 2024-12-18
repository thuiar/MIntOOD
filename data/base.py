import os
import logging
import csv
import random
import numpy as np

from .mm_pre import MMDataset, TCL_MAPDataset
from .text_pre import get_t_data
from .utils import get_v_a_data
from .text_pre import TextDataset
from .__init__ import benchmarks
from utils.functions import set_torch_seed

__all__ = ['DataManager']

class DataManager:
    
    def __init__(self, args):
        
        self.logger = logging.getLogger(args.logger_name)

        bm = benchmarks[args.dataset]
        max_seq_lengths = bm['max_seq_lengths']
        feat_dims = bm['feat_dims']

        args.label_len = bm['label_len']
        args.text_seq_len, args.video_seq_len, args.audio_seq_len  = max_seq_lengths['text'], max_seq_lengths['video'], max_seq_lengths['audio']
        args.text_feat_dim, args.video_feat_dim, args.audio_feat_dim = feat_dims['text'], feat_dims['video'], feat_dims['audio']

        if args.data_mode == 'multi-class':
            self.label_list = bm["intent_labels"]
        elif args.data_mode == 'binary-class': 
            self.label_list = bm['binary_intent_labels']
        else:
            raise ValueError('The input data mode is not supported.')
        self.logger.info('Lists of intent labels are: %s', str(self.label_list))  
        
        args.ood_label_id = len(self.label_list)
        args.num_labels = len(self.label_list) 
        
        self.data = prepare_data(args, self.logger, self.label_list, bm)
        
          
def prepare_data(args, logger, label_list, bm):          
    
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i
        
    data_path = os.path.join(args.data_path, args.dataset)

    ind_outputs = get_ind_data(args, logger, data_path, bm, label_map) 
    train_label_ids, dev_label_ids, test_label_ids = ind_outputs['train_label_ids'], ind_outputs['dev_label_ids'], ind_outputs['test_label_ids']
    text_data = ind_outputs['text_data']

    if args.method == 'text':
        
        text_train_data = TextDataset(train_label_ids, text_data['train'])
        text_dev_data = TextDataset(dev_label_ids, text_data['dev'])
        text_test_data = TextDataset(test_label_ids, text_data['test'])

        data = {'train': text_train_data, 'dev': text_dev_data, 'test': text_test_data}

    else:
        video_data, audio_data = ind_outputs['video_data'], ind_outputs['audio_data']
        if args.method == 'tcl_map':
            
            mm_train_data = TCL_MAPDataset(train_label_ids, text_data['train'], video_data['train'], audio_data['train'], text_data['train_cons_text_feats'], text_data['train_condition_idx'])
            mm_dev_data = TCL_MAPDataset(dev_label_ids, text_data['dev'], video_data['dev'], audio_data['dev'], text_data['dev_cons_text_feats'], text_data['dev_condition_idx'])
        else:
            mm_train_data = MMDataset(train_label_ids, text_data['train'], video_data['train'], audio_data['train'])
            mm_dev_data = MMDataset(dev_label_ids, text_data['dev'], video_data['dev'], audio_data['dev'])
        
        data = {'train': mm_train_data, 'dev': mm_dev_data}    
        
    if args.ood:
        
        ood_data_path = os.path.join(args.data_path, args.ood_dataset)
        ood_bm = bm['ood_data'][args.ood_dataset]
        label_map[ood_bm['ood_label']] = args.ood_label_id
        
        ood_outputs = get_ood_data(args, logger, ood_data_path, bm, label_map)

        ood_test_label_ids = ood_outputs['test_label_ids']
        test_label_ids.extend(ood_test_label_ids)

        ood_text_data = ood_outputs['text_data']
        text_data['test'].extend(ood_text_data['test'])
        text_test_data = TextDataset(test_label_ids, text_data['test'])
        
        if args.method == 'text':

            data['test'] = text_test_data

        else:

            ood_video_data, ood_audio_data = ood_outputs['video_data'], ood_outputs['audio_data']
            
            
            video_data['test']['feats'].extend(ood_video_data['test']['feats'])
            video_data['test']['lengths'].extend(ood_video_data['test']['lengths'])

            audio_data['test']['feats'].extend(ood_audio_data['test']['feats'])
            audio_data['test']['lengths'].extend(ood_audio_data['test']['lengths'])
            if args.method == 'tcl_map':
                ood_cons_text_feats, ood_condition_idx = ood_text_data['test_cons_text_feats'], ood_text_data['test_condition_idx']
                text_data['test_cons_text_feats'].extend(ood_cons_text_feats)
                text_data['test_condition_idx'].extend(ood_condition_idx)
                mm_test_data = TCL_MAPDataset(test_label_ids, text_data['test'], video_data['test'], audio_data['test'], text_data['test_cons_text_feats'], text_data['test_condition_idx'])
            else:    
                mm_test_data = MMDataset(test_label_ids, text_data['test'], video_data['test'], audio_data['test'])

            data['test'] = mm_test_data

    return data

def get_ind_data(args, logger, data_path, bm, label_map):
    logger.info('ID Data preparation...')
    if args.dataset in ['MELD-DA', 'IEMOCAP-DA', 'MIntRec2.0']:
        text_data_path = os.path.join(data_path, 'non_OOD')
    else:
        text_data_path = data_path

    train_data_index, train_label_ids = get_indexes_annotations(args, bm, label_map, os.path.join(text_data_path, 'train.tsv'), args.data_mode, args.dataset)
    dev_data_index, dev_label_ids = get_indexes_annotations(args, bm, label_map, os.path.join(text_data_path, 'dev.tsv'), args.data_mode, args.dataset)
    test_data_index, test_label_ids = get_indexes_annotations(args, bm, label_map, os.path.join(text_data_path, 'test.tsv'), args.data_mode, args.dataset)

    args.num_train_examples = len(train_data_index)
    data_args = {
        'text_data_path': text_data_path,
        'train_data_index': train_data_index,
        'dev_data_index': dev_data_index,
        'test_data_index': test_data_index,
        'bm': bm,
        'method': args.method,
    }

    text_data = get_t_data(args, data_args, ood = False)

    outputs = {}
    outputs['text_data'] = text_data
    outputs.update(data_args)

    outputs['train_label_ids'] = train_label_ids
    outputs['dev_label_ids'] = dev_label_ids
    outputs['test_label_ids'] = test_label_ids

    if args.method == 'text':
        return outputs
    else:
        video_feats_path = os.path.join(data_path, args.video_data_path, args.video_feats_path)
        video_data = get_v_a_data(data_args, video_feats_path, args.video_seq_len, ood = False)

        audio_feats_path = os.path.join(data_path, args.audio_data_path, args.audio_feats_path)
        audio_data = get_v_a_data(data_args, audio_feats_path, args.audio_seq_len, ood = False)  
        
        outputs['video_data'] = video_data
        outputs['audio_data'] = audio_data
        
        return outputs


def get_ood_data(args, logger, data_path, bm, label_map):
    
    logger.info('OOD Data preparation...')

    
    test_data_index = get_ood_indexes_annotations(os.path.join(data_path, 'test.tsv'), args.ood_dataset)

    
    test_label_ids = [args.ood_label_id] * len(test_data_index)


    data_args = {
        'text_data_path': data_path,
        'test_data_index': test_data_index,
        'bm': bm,
        'method': args.method,
    }
    text_data = get_t_data(args, data_args, ood = True)
    
    outputs = {}
    outputs['text_data'] = text_data
    outputs.update(data_args)

    outputs['test_label_ids'] = test_label_ids

    if args.method == 'text':
        return outputs

    else:

        video_feats_path = os.path.join(data_path, args.video_data_path, args.video_feats_path)
        
        video_data = get_v_a_data(data_args, video_feats_path, args.video_seq_len, ood = True)

        audio_feats_path = os.path.join(data_path, args.audio_data_path, args.audio_feats_path)
        audio_data = get_v_a_data(data_args, audio_feats_path, args.audio_seq_len, ood = True)  
        
        outputs['video_data'] = video_data
        outputs['audio_data'] = audio_data
        
        return outputs
    
def get_indexes_annotations(args, bm, label_map, read_file_path, data_mode, dataset):

    with open(read_file_path, 'r') as f:

        data = csv.reader(f, delimiter="\t")
        indexes = []
        label_ids = []

        for i, line in enumerate(data):
            if i == 0:
                continue
            
            if dataset in ['MIntRec', 'MIntRec-OOD']:
                index = '_'.join([line[0], line[1], line[2]])
                
                indexes.append(index)
            
                if data_mode == 'multi-class':
                    label_id = label_map[line[4]]
                else:
                    label_id = label_map[bm['binary_maps'][line[4]]]
            
            elif dataset in ['clinc', 'clinc-small', 'snips', 'atis']:
                label_id = label_map[line[1]]
                indexes.append(i)
                
            elif dataset in ['MELD-DA', 'MELD-DA-OOD']:
                label_id = label_map[bm['label_maps'][line[3]]]
                index = '_'.join([line[0], line[1]])
                indexes.append(index)
            
            elif dataset in ['IEMOCAP-DA', 'IEMOCAP-DA-OOD']:
                label_id = label_map[bm['label_maps'][line[2]]]
                index = line[0]
                indexes.append(index)
            
            elif dataset in ['MIntRec2.0', 'MIntRec2.0-OOD']:
                label_id = label_map[line[3]]
                index = '_'.join(['dia' + line[0], 'utt' + line[1]])
                indexes.append(index)

            label_ids.append(label_id)
    
    return indexes, label_ids


def get_ood_indexes_annotations(read_file_path, dataset):

    with open(read_file_path, 'r') as f:

        data = csv.reader(f, delimiter="\t")
        indexes = []

        for i, line in enumerate(data):
            if i == 0:
                continue
            
            if dataset in ['MIntRec', 'MIntRec-OOD', 'TED-OOD']:
                index = '_'.join([line[0], line[1], line[2]])
                
                indexes.append(index)
            
            elif dataset in ['clinc', 'clinc-small', 'snips', 'atis']:
                indexes.append(i)
                
            elif dataset in ['MELD-DA', 'MELD-DA-OOD']:
                index = '_'.join([line[0], line[1]])
                indexes.append(index)
            
            elif dataset in ['IEMOCAP-DA', 'IEMOCAP-DA-OOD']:
                index = line[0]
                indexes.append(index)
            
            elif dataset in ['MIntRec2.0', 'MIntRec2.0-OOD']:
                index = '_'.join(['dia' + line[0], 'utt' + line[1]])
                indexes.append(index)
    
    return indexes