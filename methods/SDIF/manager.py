import torch
import torch.nn.functional as F
import logging
import numpy as np
from torch import nn
from utils.functions import restore_model, save_model, EarlyStopping
from tqdm import trange, tqdm
from data.utils import get_dataloader
from utils.metrics import AverageMeter, Metrics, OOD_Metrics
from transformers import AdamW, get_linear_schedule_with_warmup
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ood_detection import ood_detection_map


class SDIF:

    def __init__(self, args, data, model):
             
        self.logger = logging.getLogger(args.logger_name)
        
        self.device = model.device
        self.model = model._set_model(args)

        mm_data = data.data
        mm_dataloader = get_dataloader(args, mm_data)
        
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            mm_dataloader['train'], mm_dataloader['dev'], mm_dataloader['test']
            
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = Metrics(args)
        self.ood_metrics = OOD_Metrics(args)
        self.ood_detection_func = ood_detection_map[args.ood_detection_method]
        
        if args.train:
            self.best_eval_score = 0
        else:
            self.model = restore_model(self.model, args.model_output_path, self.device)

    def _train(self, args): 
        
        early_stopping = EarlyStopping(args)
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=args.factor, verbose=True, patience=args.opt_patience)
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            loss_record = AverageMeter()
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)
                
                with torch.set_grad_enabled(True):

                    logits = self.model(text_feats, video_feats, audio_feats)

                    loss = self.criterion(logits, label_ids)

                    self.optimizer.zero_grad()

                    loss.backward()
                    loss_record.update(loss.item(), label_ids.size(0))
                    if args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)


                    self.optimizer.step()
                    # self.scheduler.step()
            
            outputs = self._get_outputs(args, self.eval_dataloader)
            eval_score = outputs[args.eval_monitor]
            self.scheduler.step(outputs['loss'])

            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'eval_score': round(eval_score, 4),
                'best_eval_score': round(early_stopping.best_score, 4),
            }

            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in eval_results.keys():
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            early_stopping(eval_score, self.model)

            if early_stopping.early_stop:
                self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
                break

        self.best_eval_score = early_stopping.best_score
        self.model = early_stopping.best_model   
        
        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_path)
            save_model(self.model, args.model_output_path)   

    def _get_outputs(self, args, dataloader, show_results = False, test_ind = False, flag_test=False):
        

        def capture_activation_output(module, input, output):
            global activation_output
            activation_output = output

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        total_features = torch.empty((0, args.dst_feature_dims * 3)).to(self.device)
        
        loss_record = AverageMeter()
        hook_handle = self.model.model.fusion._modules['fusion_layer_1_activation'].register_forward_hook(capture_activation_output)

        for batch in tqdm(dataloader, desc="Iteration"):

            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            
            with torch.set_grad_enabled(False):
                
                logits = self.model(text_feats, video_feats, audio_feats)
                
                total_logits = torch.cat((total_logits, logits))
                total_labels = torch.cat((total_labels, label_ids))
                total_features = torch.cat((total_features, activation_output))
                if not flag_test:
                    loss = self.criterion(logits, label_ids)
                    loss_record.update(loss.item(), label_ids.size(0))


        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)

        y_logit = total_logits.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        y_prob = total_maxprobs.cpu().numpy()
        y_feat = total_features.cpu().numpy()
        
        if test_ind:
            outputs = self.metrics(y_true[y_true != args.ood_label_id], y_pred[y_true != args.ood_label_id])
        else:
            outputs = self.metrics(y_true, y_pred, show_results = show_results)
            
        outputs.update({'loss': loss_record.avg})
        outputs.update(
            {
                'y_prob': y_prob,
                'y_logit': y_logit,
                'y_true': y_true,
                'y_pred': y_pred,
                'y_feat': y_feat
            }
        )

        return outputs

    def _test(self, args):
        
        test_results = {}
        
        ind_test_results = self._get_outputs(args, self.test_dataloader, show_results = True, test_ind = True, flag_test=True)
        if args.train:
            test_results['best_eval_score'] = round(self.best_eval_score, 4)
        test_results.update(ind_test_results)
        
        if args.ood:
            
            tmp_outputs = self._get_outputs(args, self.test_dataloader, flag_test=True)
            if args.ood_detection_method in ['residual', 'ma', 'vim']:
                ind_train_outputs = self._get_outputs(args, self.train_dataloader)
                
                tmp_outputs['train_feats'] = ind_train_outputs['y_feat']
                tmp_outputs['train_labels'] = ind_train_outputs['y_true']
                
                w, b = self.model.vim()
                tmp_outputs['w'] = w
                tmp_outputs['b'] = b
            
            scores = self.ood_detection_func(args, tmp_outputs)
            binary_labels = np.array([1 if x != args.ood_label_id else 0 for x in tmp_outputs['y_true']])
            
            ood_test_scores = self.ood_metrics(scores, binary_labels, show_results = True) 
            test_results.update(ood_test_scores)
        
        return test_results