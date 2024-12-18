import os
import csv
import sys
import torch
from transformers import BertTokenizer, RobertaTokenizer, T5Tokenizer
from torch.utils.data import Dataset
import numpy as np

def get_t_data(args, data_args, ood):
    
    if args.text_backbone.startswith('bert') or args.text_backbone.startswith('roberta') or args.text_backbone.startswith('t5'):
        t_data = get_data(args, data_args, ood)
    else:
        raise Exception('Error: inputs are not supported text backbones.')

    return t_data

def get_data(args, data_args, ood):

    processor = DatasetProcessor(args, ood = ood)
    data_path = data_args['text_data_path']

    if not ood:
        if args.method == 'tcl_map':
            train_examples = processor.get_examples(data_path, 'train') 
            train_feats, train_cons_text_feats, train_condition_idx = get_backbone_feats(args, data_args, train_examples)

            dev_examples = processor.get_examples(data_path, 'dev')
            dev_feats, dev_cons_text_feats, dev_condition_idx = get_backbone_feats(args, data_args, dev_examples)

            test_examples = processor.get_examples(data_path, 'test')
            test_feats, test_cons_text_feats, test_condition_idx = get_backbone_feats(args, data_args, test_examples)

            outputs = {
                'train': train_feats,
                'train_cons_text_feats': train_cons_text_feats,
                'train_condition_idx': train_condition_idx,
                'dev': dev_feats,
                'dev_cons_text_feats': dev_cons_text_feats,
                'dev_condition_idx': dev_condition_idx,
                'test': test_feats,
                'test_cons_text_feats': test_cons_text_feats,
                'test_condition_idx': test_condition_idx,
            }
        else:
            train_examples = processor.get_examples(data_path, 'train') 
            train_feats = get_backbone_feats(args, data_args, train_examples)

            dev_examples = processor.get_examples(data_path, 'dev')
            dev_feats = get_backbone_feats(args, data_args, dev_examples)

            test_examples = processor.get_examples(data_path, 'test')
            test_feats = get_backbone_feats(args, data_args, test_examples)

            outputs = {
                'train': train_feats,
                'dev': dev_feats,
                'test': test_feats
            }

    else:

        test_examples = processor.get_examples(data_path, 'test')
        if args.method == 'tcl_map':
            test_feats, test_cons_text_feats, test_condition_idx = get_backbone_feats(args, data_args, test_examples)
            outputs = {
                'test': test_feats,
                'test_cons_text_feats': test_cons_text_feats,
                'test_condition_idx': test_condition_idx,
            }
        else:
            test_feats = get_backbone_feats(args, data_args, test_examples)
            outputs = {
                'test': test_feats
            }

        
    return outputs

def get_backbone_feats(args, data_args, examples):
    
    if args.text_backbone.startswith('bert'):
        tokenizer = BertTokenizer.from_pretrained(args.text_pretrained_model, do_lower_case=True)   
    elif args.text_backbone.startswith('roberta'):
        tokenizer = RobertaTokenizer.from_pretrained(args.text_pretrained_model, do_lower_case=True)
    elif args.text_backbone.startswith('t5'):
        tokenizer = T5Tokenizer.from_pretrained(args.text_pretrained_model, do_lower_case=True)
    
    # <=> inputs = tokenizer(**, return_tensors="pt")
    if args.method == 'tcl_map':
        features, cons_features, condition_idx, args.max_cons_seq_length = tcl_map_convert_examples_to_features(args, examples, data_args, tokenizer)
        features_list = [[feat.input_ids, feat.input_mask, feat.segment_ids] for feat in features]
        cons_features_list = [[feat.input_ids, feat.input_mask, feat.segment_ids] for feat in cons_features]
        return features_list, cons_features_list, condition_idx
    else:
        if args.text_backbone.startswith('t5'):
            features = convert_examples_to_features_t5(examples, args.text_seq_len, tokenizer)
        else:
            features = convert_examples_to_features(examples, args.text_seq_len, tokenizer)
        features_list = [[feat.input_ids, feat.input_mask, feat.segment_ids] for feat in features]
        return features_list

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class DatasetProcessor(DataProcessor):

    def __init__(self, args, ood = False):
        super(DatasetProcessor).__init__()
        
        if not ood:
            if args.dataset in ['MIntRec']:
                self.select_id = 3
                self.label_id = 4
            elif args.dataset in ['MIntRec2.0']:
                self.select_id = 2
                self.label_id = 4
            elif args.dataset in ['MELD-DA']:
                self.select_id = 2
                self.label_id = 3
            elif args.dataset in ['IEMOCAP-DA']:
                self.select_id = 1
                self.label_id = 2
        else:
            if args.ood_dataset in ['MIntRec', 'MIntRec-OOD', 'TED-OOD']:
                self.select_id = 3
                self.label_id = 0
            elif args.ood_dataset in ['MELD-DA', 'MELD-DA-OOD', 'MIntRec2.0-OOD']:
                self.select_id = 2
                self.label_id = 0
            elif args.ood_dataset in ['IEMOCAP-DA', 'IEMOCAP-DA-OOD']:
                self.select_id = 1
                self.label_id = 0
        
    def get_examples(self, data_dir, mode):
        
        if mode == 'train':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        elif mode == 'dev':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "train")
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
        elif mode == 'all':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "all.tsv")), "all")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue

            guid = "%s-%s" % (set_type, i)
            text_a = line[self.select_id]
            if self.label_id == 0:
                label = 'OOD'
            else:
                label = line[self.label_id]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

def convert_examples_to_features_t5(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s for T5."""
    
    features = []
    for (ex_index, example) in enumerate(examples):
        # Tokenize input text
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Truncate so that the total length is less than max_seq_length
            # Account for separator token "</s>" at the end
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 1)
        else:
            # Truncate tokens_a if it exceeds max_seq_length - 1
            if len(tokens_a) > max_seq_length - 1:
                tokens_a = tokens_a[:(max_seq_length - 1)]

        # Concatenate tokens and add T5's special end token "</s>"
        tokens = tokens_a + (tokens_b if tokens_b else []) + ["</s>"]
        segment_ids = [0] * len(tokens)

        # Convert tokens to input IDs
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Create attention mask: 1 for real tokens, 0 for padding tokens
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        # Ensure the input meets the maximum sequence length requirement
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids)  # T5 does not use segment IDs
        )
    return features

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
        
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids)
                        )
    return features

def tcl_map_convert_examples_to_features(args, examples, data_args, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
        
    max_seq_length = args.text_seq_len
    label_len = data_args['bm']['label_len']
    features = []
    cons_features = []
    condition_idx = []
    prefix = ['MASK'] * 3
    # prefix = ['MASK'] * data_args['prompt_len']

    max_cons_seq_length = max_seq_length + len(prefix) + label_len
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        if args.dataset in ['MIntRec']:
            condition = tokenizer.tokenize(example.label)
        elif args.dataset in ['MELD-DA', 'IEMOCAP-DA']:
            if example.label != 'OOD':
                condition = tokenizer.tokenize(data_args['bm']['label_maps'][example.label])
            else:
                condition = tokenizer.tokenize(example.label)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # construct augmented sample pair
        cons_tokens = ["[CLS]"] + tokens_a + prefix + condition + (label_len - len(condition)) * ["MASK"] + ["[SEP]"]
        tokens = ["[CLS]"] + tokens_a + prefix + label_len * ["[MASK]"] + ["[SEP]"]

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        cons_inputs_ids = tokenizer.convert_tokens_to_ids(cons_tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_cons_seq_length - len(input_ids))
        input_ids += padding
        cons_inputs_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_cons_seq_length
        assert len(cons_inputs_ids) == max_cons_seq_length
        assert len(input_mask) == max_cons_seq_length
        assert len(segment_ids) == max_cons_seq_length
        # record the position of prompt
        condition_idx.append(1 + len(tokens_a) + len(prefix))


        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids)
                        )
        
        cons_features.append(
            InputFeatures(input_ids=cons_inputs_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids)
                        )
    return features, cons_features, condition_idx, max_cons_seq_length


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  # For dialogue context
        else:
            tokens_b.pop()

class TextDataset(Dataset):
    
    def __init__(self, label_ids, text_feats):
        
        self.label_ids = torch.tensor(label_ids)
        self.text_feats = torch.tensor(text_feats)
        self.size = len(self.text_feats)

    def __len__(self):
        return self.size

    def __getitem__(self, index):

        sample = {
            'text_feats': self.text_feats[index],
            'label_ids': self.label_ids[index], 
        } 
        return sample