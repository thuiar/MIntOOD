class Param():
    
    def __init__(self, args):
        
        self.hyper_param = self._get_hyper_parameters(args)
        
    def _get_hyper_parameters(self, args):
        """
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
        """
        ood_detection_parameters = {
            'sbm':{
                'temperature': [1e4],
                'scale': [20]
            }
        }
        if args.text_backbone.startswith('roberta'):

            hyper_parameters = {
                'eval_monitor': ['f1'],
                'train_batch_size': 16,
                'eval_batch_size': 8,
                'test_batch_size': 8,
                'wait_patience': 8,
                'num_train_epochs': 100,
                ################
                'warmup_proportion': [0.1],
                'lr': [2e-5],
                'weight_decay': [0.03],
                'scale':32
            } 
        else:
            raise ValueError('Not supported text backbone')        
        
        if args.ood_detection_method in ood_detection_parameters.keys():
            ood_parameters = ood_detection_parameters[args.ood_detection_method]
            hyper_parameters.update(ood_parameters)
            
        return hyper_parameters 