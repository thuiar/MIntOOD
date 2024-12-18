class Param():
    
    def __init__(self, args):

        self.hyper_param = self._get_hyper_parameters(args)

    def _get_hyper_parameters(self, args):
        '''
        Args:
            num_train_epochs (int): The number of training epochs.
            num_labels (autofill): The output dimension.
            max_seq_length (autofill): The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
            freeze_backbone_parameters (binary): Whether to freeze all parameters but the last layer.
            feat_dim (int): The feature dimension.
            warmup_proportion (float): The warmup ratio for learning rate.
            activation (str): The activation function of the hidden layer (support 'relu' and 'tanh').
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            test_batch_size (int): The batch size for testing.
            wait_patient (int): Patient steps for Early Stop.
        '''
        if args.text_backbone.startswith('bert'):

            hyper_parameters = {
                'data_mode': 'multi-class',
                'padding_mode': 'zero',
                'padding_loc': 'end',
                'need_aligned': False,
                'eval_monitor': ['f1'],
                'train_batch_size': [16],
                'eval_batch_size': 8,
                'test_batch_size': 8,
                'wait_patience': [8],
                'num_train_epochs': [100],
                'dst_feature_dims': 768,
                'n_levels_self': 1,
                'n_levels_cross': 1,
                'dropout_rate': 0.2,
                'cross_dp_rate': 0.3,
                'cross_num_heads': 12,
                'self_num_heads': 8,
                'grad_clip': 7, 
                'lr': [9e-6], #9e-6原始
                'opt_patience': 8,
                'factor': 0.5,
                'weight_decay': 0.01,
                'aug_lr': 1e-6, #1e-6原始
                'aug_epoch': 1,
                'aug_dp': 0.3,
                'aug_weight_decay': 0.1,
                'aug_grad_clip': -1.0,
                'aug_batch_size': 16,
                'aug': True,
                'aug_num': 25000,
                'use_wandb': False,
                'scale': [16],
            }
            # hyper_parameters = {
            #     'need_aligned': True,
            #     'freeze_parameters': False,
            #     'eval_monitor': 'f1',
            #     'eval_batch_size': 16,
            #     'wait_patience': [8],
            #     'binary_multiple_ood': 1.0, 
            #     'base_dim': [768],
            #     'lr': [3e-5], #3e-5
            #     'temperature': [2], # bigger is usually better
            #     'alpha': [2], #0.5, 1
            #     'mlp_hidden_size': [256],
            #     'mlp_dropout': [0.1],
            #     're_prob': [0.1],
            #     'num_train_epochs': [100], # [30, 40, 50]
            #     'train_batch_size': [32], # [32, 64, 128]
            #     'weight_decay': [0.01], # [0.01, 0.05, 0.1]
            #     'multiple_ood': [1.0], # try average number
            #     'contrast_dropout': [0.1],
            #     'select_number_min': [2],
            #     'select_number_max': [3],
            #     'weight_dropout': [0.5],
            #     'weight_hidden_dim': [256],
            #     # 'weight': [2, 3],
            #     'aligned_method': ['ctc'],
            #     'warmup_proportion': [0.1], 
            #     'scale': [16],
            #     'encoder_layers_a': [1],
            #     'encoder_layers_v': [2],
            #     'attn_dropout': [0.0],
            #     'relu_dropout': [0.1],    
            #     'embed_dropout': [0.0],
            #     'res_dropout': [0.2],   #0
            #     'attn_mask': [False],   #True
            #     'nheads': [2], #4
            # }

        return hyper_parameters