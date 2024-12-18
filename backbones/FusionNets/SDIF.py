import torch
from ..SubNets.FeatureNets import BERTEncoder, BertCrossEncoder, BERTEncoderSDIF
from ..SubNets import text_backbones_map
from torch import nn

__all__ = ['SDIF']


class SDIF(nn.Module):
    
    def __init__(self, args):

        super(SDIF, self).__init__()
        self.args = args
        

        self.text_subnet = BERTEncoderSDIF.from_pretrained(args.text_pretrained_model)
        self.visual_size = args.video_feat_dim
        self.acoustic_size = args.audio_feat_dim
        self.text_size = args.text_feat_dim
        self.device = args.device
        self.dst_feature_dims = args.dst_feature_dims
        # self.capture_activation_output = capture_activation_output

        self.layers_cross = args.n_levels_cross 
        self.layers_self = args.n_levels_self 

        self.dropout_rate = args.dropout_rate
        self.cross_dp_rate = args.cross_dp_rate
        self.cross_num_heads = args.cross_num_heads
        self.self_num_heads = args.self_num_heads

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dst_feature_dims, nhead=self.self_num_heads)
        self.self_att = nn.TransformerEncoder(encoder_layer, num_layers=self.layers_self)

        self.video2text_cross = BertCrossEncoder(self.cross_num_heads, self.dst_feature_dims, self.cross_dp_rate, n_layers=self.layers_cross)
        self.audio2text_cross = BertCrossEncoder(self.cross_num_heads, self.dst_feature_dims, self.cross_dp_rate, n_layers=self.layers_cross)
        
        self.v2t_project = nn.Linear(self.visual_size, self.text_size)

        self.mlp_project =  nn.Sequential(
                nn.Linear(self.dst_feature_dims, self.dst_feature_dims),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )
        
        self.shallow_att_project = nn.Linear(self.dst_feature_dims, 1, bias=False)
        self.deep_att_project = nn.Linear(self.dst_feature_dims, 1, bias=False)

        self.activation = nn.ReLU()
        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.dst_feature_dims * 6, out_features=self.dst_feature_dims * 3))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(self.dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=self.dst_feature_dims * 3, out_features= args.num_labels))

        if args.aug:
            self.out_layer = nn.Linear(self.dst_feature_dims, args.num_labels)
            self.aug_dp = nn.Dropout(args.aug_dp)

    def forward(self, text_feats, video_feats, audio_feats, pre_train = False):

        if pre_train:
            text_outputs = self.text_subnet(text_feats)
            text_seq, text_rep = text_outputs['last_hidden_state'], text_outputs['pooler_output']
            text_rep = self.aug_dp(text_rep)
            logits = self.out_layer(text_rep)
            return logits
        
        # first layer : T,V,A
        bert_sent, bert_sent_mask, bert_sent_type = text_feats[:,0], text_feats[:,1], text_feats[:,2]
        text_outputs = self.text_subnet(text_feats)
        text_seq, text_rep = text_outputs['last_hidden_state'], text_outputs['pooler_output']
        
        video_feats = video_feats.to(self.v2t_project.weight.dtype)
        video_seq = self.v2t_project(video_feats)
        audio_seq = audio_feats

        video_mask = torch.sum(video_feats.ne(torch.zeros(video_feats[0].shape[-1]).to(self.device)).int(), dim=-1)/video_feats[0].shape[-1]
        video_mask_len = torch.sum(video_mask, dim=1, keepdim=True)  

        video_mask_len = torch.where(video_mask_len > 0.5, video_mask_len, torch.ones([1]).to(self.device))
        video_masked_output = torch.mul(video_mask.unsqueeze(2), video_seq)
        video_rep = torch.sum(video_masked_output, dim=1, keepdim=False) / video_mask_len
        

        audio_mask = torch.sum(audio_feats.ne(torch.zeros(audio_feats[0].shape[-1]).to(self.device)).int(), dim=-1)/audio_feats[0].shape[-1]
        audio_mask_len = torch.sum(audio_mask, dim=1, keepdim=True)  
        
        audio_masked_output = torch.mul(audio_mask.unsqueeze(2), audio_seq)
        audio_rep = torch.sum(audio_masked_output, dim=1, keepdim=False) / audio_mask_len
        

        # Second layer (V,A) --> T: V_T, A_T
        extended_video_mask = video_mask.unsqueeze(1).unsqueeze(2)
        extended_video_mask = extended_video_mask.to(dtype=next(self.parameters()).dtype)
        extended_video_mask = (1.0 - extended_video_mask) * -10000.0
        video2text_seq = self.video2text_cross(text_seq, video_seq, extended_video_mask)

        extended_audio_mask = audio_mask.unsqueeze(1).unsqueeze(2)
        extended_audio_mask = extended_audio_mask.to(dtype=next(self.parameters()).dtype)
        extended_audio_mask = (1.0 - extended_audio_mask) * -10000.0
        audio2text_seq = self.audio2text_cross(text_seq, audio_seq, extended_audio_mask)

        text_mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True) 

        video2text_masked_output = torch.mul(bert_sent_mask.unsqueeze(2), video2text_seq)
        video2text_rep = torch.sum(video2text_masked_output, dim=1, keepdim=False) / text_mask_len
        

        audio2text_masked_output = torch.mul(bert_sent_mask.unsqueeze(2), audio2text_seq)
        audio2text_rep = torch.sum(audio2text_masked_output, dim=1, keepdim=False) / text_mask_len
        

        # Third layer: mlp->VAL
        shallow_seq = self.mlp_project(torch.cat([audio2text_seq, text_seq, video2text_seq], dim=1))
        
        # Deep Interaction
        tri_cat_mask = torch.cat([bert_sent_mask, bert_sent_mask, bert_sent_mask], dim=-1)

        tri_mask_len = torch.sum(tri_cat_mask, dim=1, keepdim=True) 
        shallow_masked_output = torch.mul(tri_cat_mask.unsqueeze(2), shallow_seq)
        shallow_rep = torch.sum(shallow_masked_output, dim=1, keepdim=False) / tri_mask_len

        text_rep = text_rep.to(video2text_rep.dtype)
        video_rep = video_rep.to(video2text_rep.dtype)
        audio_rep = audio_rep.to(video2text_rep.dtype)
        audio2text_rep = audio2text_rep.to(video2text_rep.dtype)

        all_reps = torch.stack((text_rep, video_rep, audio_rep, video2text_rep, audio2text_rep, shallow_rep), dim=0)
        all_hiddens = self.self_att(all_reps)
        deep_rep = torch.cat((all_hiddens[0], all_hiddens[1], all_hiddens[2], all_hiddens[3], all_hiddens[4], all_hiddens[5]), dim=1)

        self.text_rep = text_rep
        self.video_rep = video_rep
        self.audio_rep = audio_rep
        self.video2text_rep = video2text_rep
        self.audio2text_rep = audio2text_rep
        self.shallow_rep = shallow_rep


        logits = self.fusion(deep_rep)
        
        return logits
    
    def vim(self):

        return self.fusion.fusion_layer_3.weight, self.fusion.fusion_layer_3.bias


    

