import torch.nn.functional as F
import torch
import torch.utils.checkpoint
import math
import numpy as np
from torch import nn
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from ..SubNets.AlignNets import AlignSubNet
from torch.nn.parameter import Parameter
from ..SubNets.transformers_encoder.transformer import TransformerEncoder

class Fusion_ADD(nn.Module):
    def __init__(self,  config, args):
        super(Fusion_ADD, self).__init__()
        self.args = args

        text_feat_dim, audio_feat_dim, video_feat_dim = args.text_feat_dim, args.audio_feat_dim, args.video_feat_dim

        self.num_heads = args.nheads
        self.attn_dropout = args.attn_dropout
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.attn_mask = args.attn_mask

        self.audio_proj = nn.Linear(audio_feat_dim, args.base_dim)
        self.video_proj = nn.Linear(video_feat_dim, args.base_dim)
        
        self.v_encoder = self.get_transformer_encoder(video_feat_dim, args.encoder_layers_v)
        self.a_encoder = self.get_transformer_encoder(audio_feat_dim, args.encoder_layers_a)

    def forward(self, textual, visual, acoustic, labels = None, sampler = None, attention_mask = None, ood_sampling = False, data_aug = False, probs = None):
        
    
        visual = visual.permute(1, 0, 2)
        visual = self.v_encoder(visual).mean(dim = 0)
        visual = self.video_proj(visual)

        acoustic = acoustic.permute(1, 0, 2)
        acoustic = self.a_encoder(acoustic).mean(dim = 0)
        acoustic = self.audio_proj(acoustic)

        textual = textual[:, 0]
        
        fusion = textual + visual + acoustic

        return textual, visual, acoustic, fusion

    def get_transformer_encoder(self, embed_dim, layers):
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=layers,
                                  attn_dropout=self.attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

class Fusion_CONCAT(nn.Module):
    def __init__(self,  config, args):
        super(Fusion_CONCAT, self).__init__()
        self.args = args

        text_feat_dim, audio_feat_dim, video_feat_dim = args.text_feat_dim, args.audio_feat_dim, args.video_feat_dim

        self.num_heads = args.nheads
        self.attn_dropout = args.attn_dropout
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.attn_mask = args.attn_mask

        self.audio_proj = nn.Linear(audio_feat_dim, args.base_dim)
        self.video_proj = nn.Linear(video_feat_dim, args.base_dim)
        
        self.v_encoder = self.get_transformer_encoder(video_feat_dim, args.encoder_layers_v)
        self.a_encoder = self.get_transformer_encoder(audio_feat_dim, args.encoder_layers_a)

        self.fusion_layer = nn.Linear(3 * args.base_dim, args.base_dim)

    def forward(self, textual, visual, acoustic, labels = None, sampler = None, attention_mask = None, ood_sampling = False, data_aug = False, probs = None):
        
    
        visual = visual.permute(1, 0, 2)
        visual = self.v_encoder(visual).mean(dim = 0)
        visual = self.video_proj(visual)

        acoustic = acoustic.permute(1, 0, 2)
        acoustic = self.a_encoder(acoustic).mean(dim = 0)
        acoustic = self.audio_proj(acoustic)

        textual = textual[:, 0]
        
        fusion = self.fusion_layer(torch.cat([textual, visual, acoustic], dim=1))

        return textual, visual, acoustic, fusion

    def get_transformer_encoder(self, embed_dim, layers):
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=layers,
                                  attn_dropout=self.attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

class Fusion_WEIGHT(nn.Module):
    def __init__(self,  config, args):
        super(Fusion_WEIGHT, self).__init__()
        self.args = args

        text_feat_dim, audio_feat_dim, video_feat_dim = args.text_feat_dim, args.audio_feat_dim, args.video_feat_dim

        self.num_heads = args.nheads
        self.attn_dropout = args.attn_dropout
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.attn_mask = args.attn_mask

        self.audio_proj = nn.Linear(audio_feat_dim, args.base_dim)
        self.video_proj = nn.Linear(video_feat_dim, args.base_dim)
     
        self.v_encoder = self.get_transformer_encoder(video_feat_dim, args.encoder_layers_v)
        self.a_encoder = self.get_transformer_encoder(audio_feat_dim, args.encoder_layers_a)

        self.text_weight_net = nn.Sequential(
            nn.Linear(args.base_dim, args.weight_hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.weight_dropout),
            nn.Linear(args.weight_hidden_dim, 1),
        )
        self.video_weight_net = nn.Sequential(
            nn.Linear(args.base_dim, args.weight_hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.weight_dropout),
            nn.Linear(args.weight_hidden_dim, 1),
        )
        self.audio_weight_net = nn.Sequential(
            nn.Linear(args.base_dim, args.weight_hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.weight_dropout),
            nn.Linear(args.weight_hidden_dim, 1),
        )

    def forward(self, textual, visual, acoustic, labels = None, sampler = None, attention_mask = None, ood_sampling = False, data_aug = False, probs = None):
        
    
        visual = visual.permute(1, 0, 2)
        visual = self.v_encoder(visual).mean(dim = 0)
        visual = self.video_proj(visual)

        acoustic = acoustic.permute(1, 0, 2)
        acoustic = self.a_encoder(acoustic).mean(dim = 0)
        acoustic = self.audio_proj(acoustic)

        textual = textual[:, 0]

        text_weight = self.text_weight_net(textual)
        video_weight = self.video_weight_net(visual)
        audio_weight = self.audio_weight_net(acoustic)
        
        weights = torch.cat([text_weight, video_weight, audio_weight], dim=1)
        normalized_weights = F.softmax(weights, dim=1)

        weights_filter = ~(attention_mask.sum(dim=1) == 2).unsqueeze(1) # remove non-text sample
        if text_weight.shape[0] > weights_filter.shape[0]:
            weights_filter = torch.cat([weights_filter, weights_filter], dim=0)
        
        va_weights = torch.cat([video_weight, audio_weight], dim=1)
        va_normalized_weights = F.softmax(va_weights, dim=1)
        text_weight_pading = torch.zeros_like(text_weight).to(self.args.device)
        
        new_normalized_weights = torch.cat([text_weight_pading, va_normalized_weights], dim=1)
        selected_weights = torch.where(weights_filter, normalized_weights, new_normalized_weights)
        
        
        fusion = selected_weights[:, 0:1] * textual + \
                 selected_weights[:, 1:2] * visual + \
                 selected_weights[:, 2:3] * acoustic

        return textual, visual, acoustic, fusion

    def get_transformer_encoder(self, embed_dim, layers):
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=layers,
                                  attn_dropout=self.attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

class Fusion_BertModel(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.config = config
        self.args = args

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        sampler,
        labels,
        ood_sampling,
        data_aug,
        probs,
        binary, 
        ood_elems,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        if ood_sampling:
            mix_data, mix_labels = sampler(embedding_output, visual, acoustic, labels, extended_attention_mask, attention_mask, self.args.device, binary = binary, ood_elems = ood_elems)
            embedding_output, visual, acoustic, extended_attention_mask, attention_mask = mix_data['text'], mix_data['video'], mix_data['audio'], mix_data['mask'], mix_data['attention_mask']

     
        text_encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        text_sequence_output = text_encoder_outputs[0]
        
        if ood_sampling:
            return text_sequence_output, visual, acoustic, mix_labels
        else:
            return text_sequence_output, visual, acoustic

class Fusion_Model(BertPreTrainedModel):

    def __init__(self, config, args):

        super().__init__(config)
        self.num_labels = args.num_labels

        self.bert = Fusion_BertModel(config, args)

        self.args = args

        if args.ablation_type == 'fusion_add':
            self.fusion = Fusion_ADD(config, args)
        elif args.ablation_type == 'fusion_concat':
            self.fusion = Fusion_CONCAT(config, args)
        else:
            self.fusion = Fusion_WEIGHT(config, args)
        
        self.init_weights()

    def forward(
        self,
        text,
        visual,
        acoustic,
        sampler,
        labels,
        ood_sampling,
        data_aug,
        probs,
        binary,
        ood_elems,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
   
        input_ids, attention_mask, token_type_ids = text[:, 0], text[:, 1], text[:, 2]

        if ood_sampling:
            # attention_mask = torch.cat((attention_mask, attention_mask), dim=0)
            # token_type_ids = torch.cat((token_type_ids, token_type_ids), dim=0)
            pooled_output, v, a, mixed_labels = self.bert(
                input_ids,
                visual,
                acoustic,
                sampler,
                labels,
                ood_sampling,
                data_aug,
                probs,
                binary,
                ood_elems,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        else:
            pooled_output, v, a = self.bert(
                input_ids,
                visual,
                acoustic,
                sampler,
                labels,
                ood_sampling,
                data_aug,
                probs,
                binary,
                ood_elems,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        
        textual, visual, acoustic, pooled_output  = self.fusion(pooled_output, v, a, attention_mask = attention_mask, ood_sampling = ood_sampling, sampler = sampler, data_aug = data_aug, labels = labels, probs = probs)
        # pooled_output = pooled_output.mean(dim=1)
        if self.args.ablation_type == 'text':
            if ood_sampling:
                return textual, mixed_labels
            else:
                return textual
        else:
            if ood_sampling:
                return pooled_output, mixed_labels
            else:
                return pooled_output

    def classify(self, dropped_output):

        logits = self.classifier(dropped_output)

        return logits

    def vim(self):

        return self.classifier.weight, self.classifier.bias

class Fusion(nn.Module):
    def __init__(self, args):

        super(Fusion, self).__init__()

        self.model = Fusion_Model.from_pretrained(
            args.text_pretrained_model, args=args)

        args.feat_size = args.text_feat_dim
        
    def forward(self, text_feats, video_feats, audio_feats, sampler, labels = None, ood_sampling = False, data_aug = False, probs=None, binary = False, ood_elems = None):
        
        video_feats, audio_feats = video_feats.float(), audio_feats.float()
        
        if ood_sampling: 
            pooled_output, mixed_labels = self.model(
                text=text_feats,
                visual=video_feats,
                acoustic=audio_feats,
                sampler=sampler,
                labels=labels,
                ood_sampling=ood_sampling,
                data_aug = data_aug,
                probs=probs,
                binary = binary,
                ood_elems = ood_elems
            )
        else:
            pooled_output = self.model(
                text=text_feats,
                visual=video_feats,
                acoustic=audio_feats,
                sampler=sampler,
                labels=labels,
                ood_sampling=ood_sampling,
                data_aug = data_aug,
                probs=probs,
                binary = binary,
                ood_elems = ood_elems
            )

        if ood_sampling: 
            return pooled_output, mixed_labels
        else:
            return pooled_output

    def vim(self):

        return self.model.vim()