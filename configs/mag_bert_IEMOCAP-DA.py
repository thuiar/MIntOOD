# class Param():
    
#     def __init__(self, args):
        
#         self.common_param = self._get_common_parameters(args)
#         self.hyper_param = self._get_hyper_parameters(args)

#     def _get_common_parameters(self, args):
#         """
#             padding_mode (str): The mode for sequence padding ('zero' or 'normal').
#             padding_loc (str): The location for sequence padding ('start' or 'end'). 
#             eval_monitor (str): The monitor for evaluation ('loss' or metrics, e.g., 'f1', 'acc', 'precision', 'recall').  
#             need_aligned: (bool): Whether to perform data alignment between different modalities.
#             train_batch_size (int): The batch size for training.
#             eval_batch_size (int): The batch size for evaluation. 
#             test_batch_size (int): The batch size for testing.
#             wait_patience (int): Patient steps for Early Stop.
#         """
#         common_parameters = {
#             'padding_mode': 'zero',
#             'padding_loc': 'end',
#             'need_aligned': False,
#             'eval_monitor': 'f1',
#             'train_batch_size': 16,
#             'eval_batch_size': 8,
#             'test_batch_size': 8,
#             'wait_patience': 8
#         }
#         return common_parameters

#     def _get_hyper_parameters(self, args):
#         """
#         Args:
#             num_train_epochs (int): The number of training epochs.
#             dst_feature_dims (int): The destination dimensions (assume d(l) = d(v) = d(t)).
#             nheads (int): The number of heads for the transformer network.
#             n_levels (int): The number of layers in the network.
#             attn_dropout (float): The attention dropout.
#             attn_dropout_v (float): The attention dropout for the video modality.
#             attn_dropout_a (float): The attention dropout for the audio modality.
#             relu_dropout (float): The relu dropout.
#             embed_dropout (float): The embedding dropout.
#             res_dropout (float): The residual block dropout.
#             output_dropout (float): The output layer dropout.
#             text_dropout (float): The dropout for text features.
#             grad_clip (float): The gradient clip value.
#             attn_mask (bool): Whether to use attention mask for Transformer. 
#             conv1d_kernel_size_l (int): The kernel size for temporal convolutional layers (text modality).  
#             conv1d_kernel_size_v (int):  The kernel size for temporal convolutional layers (video modality).  
#             conv1d_kernel_size_a (int):  The kernel size for temporal convolutional layers (audio modality).  
#             lr (float): The learning rate of backbone.
#         """
#         hyper_parameters = {
#             'num_train_epochs': 100,
#             'dst_feature_dims': 120,
#             'nheads': 8,
#             'n_levels': 8,
#             'attn_dropout': 0.0,
#             'attn_dropout_v': 0.2,
#             'attn_dropout_a': 0.2,
#             'relu_dropout': 0.0,
#             'embed_dropout': 0.1,
#             'res_dropout': 0.0,
#             'output_dropout': 0.2,
#             'text_dropout': 0.4,
#             'grad_clip': 0.5, 
#             'attn_mask': True,
#             'conv1d_kernel_size_l': 5,
#             'conv1d_kernel_size_v': 1,
#             'conv1d_kernel_size_a': 1,
#             'lr': 0.00003,
#         }
#         return hyper_parameters
        
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
        if args.text_backbone.startswith('bert'):

            hyper_parameters = {
                'need_aligned': True,
                'eval_monitor': ['f1'],
                'train_batch_size': 16,
                'eval_batch_size': 8,
                'test_batch_size': 8,
                'wait_patience': 8,
                'num_train_epochs': 100,
                ################
                'beta_shift': 0.005,
                'dropout_prob': 0.5,
                'warmup_proportion': 0.1,
                'lr': 2e-5,
                'aligned_method': 'ctc',
                'weight_decay': 0.03,
                'scale':32

            } 
        else:
            raise ValueError('Not supported text backbone')        
        
        if args.ood_detection_method in ood_detection_parameters.keys():
            ood_parameters = ood_detection_parameters[args.ood_detection_method]
            hyper_parameters.update(ood_parameters)
            
        return hyper_parameters 