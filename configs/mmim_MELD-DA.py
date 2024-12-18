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
        ood_detection_parameters = {
            'sbm':{
                'temperature': [1e6],
                'scale': [20]
            },
            'hub':{
                'temperature': [1e6],
                'scale': [20],
                'k': [10],
                'alpha': [0.5]
            }
        }
        if args.text_backbone.startswith('bert'):
            hyper_parameters = {
                'need_aligned': False,
                'eval_monitor': ['f1'],
                'train_batch_size': 16,
                'eval_batch_size': 8,
                'test_batch_size': 8,
                'wait_patience': 8,
                'num_train_epochs': 100,
                #####################
                'gamma': [1],
                'theta': [1],
                'add_va': False,
                'cpc_activation': 'Tanh',
                'mmilb_mid_activation': 'ReLU',
                'mmilb_last_activation': 'Tanh',
                'optim': 'Adam',
                'contrast': True,
                'bidirectional': True,
                'grad_clip': [0.9],
                'lr_main': [2e-5],
                'weight_decay_main': [0.0001],
                'lr_bert': [2e-5],
                'weight_decay_bert': [0.001],
                'lr_mmilb': [0.001],
                'weight_decay_mmilb': [0.0004],
                'alpha': [0.4],
                'beta': [0.1],
                'dropout_a': [0.2],   #0.1
                'dropout_v': [0.2],  #0.1
                'dropout_prj': [0.2],
                'n_layer': [2],
                'cpc_layers': [4],
                'd_vh': [8],
                'd_ah': [4],
                'd_vout': [4],
                'd_aout': [4],
                'd_prjh': [512],
                'scale': [20]
            }
        # if args.ood_detection_method in ood_detection_parameters.keys():   
        #     ood_parameters = ood_detection_parameters[args.ood_detection_method]
        #     hyper_parameters.update(ood_parameters)
            
        return hyper_parameters