import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from .wavlm import WavLMMAMHead, WavLMForMultiTurn
from transformers.models.roberta.modeling_roberta import RobertaLMHead, RobertaModel, RobertaLayer
from transformers import PretrainedConfig, WavLMConfig, RobertaConfig
import os

CUSTOM_CONFIG_NAME = "config.json"
AUDIO_CONFIG_NAME = "audio_config.json"
TEXT_CONFIG_NAME = "text_config.json"

class ATConfig(PretrainedConfig):
    audio_config_cls = WavLMConfig
    text_config_cls = RobertaConfig

    def __init__(self):
        super().__init__()
        self.text = self.text_config_cls()
        self.audio = self.audio_config_cls()

    def save_pretrained(self, save_directory, push_to_hub=False, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.audio.to_json_file(os.path.join(save_directory, AUDIO_CONFIG_NAME), True)
        self.text.to_json_file(os.path.join(save_directory, TEXT_CONFIG_NAME), True)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs):
        config = cls.from_json_files(os.path.join(pretrained_model_name_or_path, AUDIO_CONFIG_NAME),
                                     os.path.join(pretrained_model_name_or_path, TEXT_CONFIG_NAME))
        if not return_unused_kwargs or len(kwargs) == 0:
            return config
        return config, kwargs

    @classmethod
    def from_configs(cls, audio, text):
        config = cls()
        config.audio = audio
        config.text = text
        return config

    @classmethod
    def from_classes(cls, audio, text):
        return cls.from_configs(cls.audio_config_cls.from_pretrained(audio), cls.text_config_cls.pretrained(text))

    @classmethod
    def from_json_files(cls, audio, text):
        return cls.from_configs(cls.audio_config_cls.from_json_file(audio), cls.text_config_cls.from_json_file(text))

    def set_pooling_mode(self, audio, text):
        self.text.pooling_mode = text
        self.audio.pooling_mode = audio

    def set_length(self, audio, text):
        self.text.max_length = text
        self.audio.max_length = audio


class ATModel(PreTrainedModel):
    config_class = ATConfig
    _keys_to_ignore_on_load_missing = ["mlm_head", "mam_head", "selection_head", "start_prediction_head", "end_prediction_head"]
    _keys_to_ignore_on_save = ["mlm_head", "mam_head", "selection_head", "start_prediction_head", "end_prediction_head"]

    def __init__(self, config: ATConfig, audio=None, text=None):
        super(ATModel, self).__init__(config)
        self.hidden_size = config.text.hidden_size
        if audio is None:
            self.audio_encoder = WavLMForMultiTurn(config.audio)
            self.text_encoder = RobertaModel(config.text)
        else:
            self.audio_encoder = WavLMForMultiTurn.from_pretrained(audio, config=config.audio)
            self.text_encoder = RobertaModel.from_pretrained(text, config=config.text)
        self.token_type_embeddings = nn.Embedding(2, self.hidden_size)
        self.fused_encoder = nn.ModuleList(RobertaLayer(config.text) for _ in range(config.text.num_fused_layers))

    def fuse_four(self, text, audio, bs, text_len, audio_len, token_type_ids=None):
        text = text.unsqueeze(2).repeat(1, 1, 2, 1, 1).view(4 * bs, text_len, -1)
        audio = audio.unsqueeze(1).repeat(1, 2, 1, 1, 1).view(4 * bs, audio_len, -1)
        fused_input = torch.cat([text, audio], dim=1)
        if token_type_ids is not None:
            fused_input += self.token_type_embeddings(token_type_ids)
        else:
            fused_input = fused_input.squeeze(-1)
        return fused_input

    def forward(self, audio_input, text_input, audio_attention_mask, text_attention_mask, bs, turn_id=None):
        device = audio_input.device
        # audio: 3B * 160000  text: 2B * 514  mlm_label: B * 514  turn_id: B * 514
        out = self.audio_encoder(audio_input, audio_attention_mask, bs, perform_mam=True,
                                 token_embedding=self.text_encoder.embeddings.token_type_embeddings)
        audio_features, audio_mask, mam_label, a_masked = out
        # audio_features: 2B * 200 * 768  audio_mask: 2B * 200  mam_label: B * 200  a_masked: B * 200
        text_features = self.text_encoder(text_input, text_attention_mask, token_type_ids=turn_id)[0]
        # text_features: 2B * 514 * 768
        bs, text_len = text_input.shape
        bs //= 2
        audio_features = audio_features.view(bs, 2, -1, self.hidden_size)
        text_features = text_features.view(bs, 2, text_len, self.hidden_size)
        audio_len = audio_features.shape[2]
        modal_ids = torch.zeros([bs * 4, text_len + audio_len], dtype=torch.long).to(device)
        modal_ids[:, text_len:] = 1
        fused_input = self.fuse_four(text_features, audio_features, bs, text_len, audio_len, modal_ids)
        fused_attention_mask = self.fuse_four(text_attention_mask, audio_mask, bs, text_len, audio_len)
        fused_attention_mask = (1.0 - fused_attention_mask[:, None, None, :]) * torch.finfo(text_features.dtype).min
        for layer in self.fused_encoder:
            fused_input = layer(fused_input, fused_attention_mask)[0]
        return fused_input, mam_label, a_masked


class ATForSequenceClassification(PreTrainedModel):
    config_class = ATConfig
    _keys_to_ignore_on_load_missing = ["fc", "activation", "cls_head"]

    def __init__(self, config: ATConfig, args, audio=None, text=None):
        super(ATForSequenceClassification, self).__init__(config)
        self.num_labels = args.num_labels
        self.hidden_size = config.text.hidden_size
        self.model = ATModel(config, audio, text)
        hidden_size = config.text.hidden_size
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.activation = ACT2FN['gelu']
        self.cls_head = nn.Linear(hidden_size, self.num_labels)
        # self.head = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size),
        #     ACT2FN['gelu'],
        #     nn.Linear(hidden_size, self.num_labels))
        self.config = config

    def forward(self, text, audio):
        text_input, text_attention_mask, token_type_ids = text[:, 0], text[:, 1], text[:, 2]
        audio_input, audio_attention_mask = audio[:, 0], audio[:, 1]
        bs, text_len = text_input.shape[:2]
        # audio_features, audio_mask = self.model.audio_encoder(audio_input, audio_attention_mask, bs, False, self.model.text_encoder.embeddings.token_type_embeddings if self.config.audio.multi_turn else None)
        audio_features, audio_mask = self.model.audio_encoder(audio_input, audio_attention_mask, bs, False, None)
        text_features = self.model.text_encoder(text_input, text_attention_mask, token_type_ids=token_type_ids)[0]
        modal_ids = torch.zeros([bs, text_len + audio_features.shape[1]], dtype=torch.long).to(text_input.device)
        modal_ids[:, text_len:] = 1
        fused_input = torch.cat([text_features, audio_features], dim=1) + self.model.token_type_embeddings(modal_ids)
        fused_attention_mask = torch.cat([text_attention_mask, audio_mask], dim=1).to(dtype=text_features.dtype)
        fused_attention_mask = (1.0 - fused_attention_mask[:, None, None, :]).to(dtype=text_features.dtype) * torch.finfo(text_features.dtype).min
        if hasattr(self.config.text, "num_fusion_layers") or hasattr(self.config.text, "num_fused_layers"):
            for layer in self.model.fused_encoder:
                fused_input = layer(fused_input, fused_attention_mask)[0]
        else:
            fused_input = self.model.fused_encoder(fused_input, fused_attention_mask)[0]
        fused_input = fused_input[:, 0]
        feats = self.activation(self.fc(fused_input))
        logits = self.cls_head(feats).squeeze(1)

        return logits, feats

    def vim(self):
        return self.cls_head.weight, self.cls_head.bias

    def _init_weights(self, module):
        pass


class Spectra(nn.Module):
    def __init__(self, args):
        super(Spectra, self).__init__()
        config = ATConfig.from_pretrained(args.spectra_path)
        self.model = ATForSequenceClassification.from_pretrained(args.spectra_path, config=config, args=args)

    def forward(self, text, video, audio):
        return self.model(text, audio)
    
    def vim(self):
        return self.model.vim()
