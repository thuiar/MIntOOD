# Multimodal Classification and Out-of-distribution Detection for Multimodal Intent Understanding

## 1. Introduction

This repository contains the official PyTorch implementation of the research paper [Multimodal Classification and Out-of-distribution Detection for Multimodal Intent Understanding](https://arxiv.org/abs/2412.12453). 

Multimodal intent understanding is a substantial field that needs to utilize nonverbal modalities effectively in analyzing human language. However, there is a lack of research focused on adapting these methods to real-world scenarios with out-of-distribution (OOD) samples, a key aspect in creating robust and secure systems. In this paper, we propose MIntOOD, a module for OOD detection in multimodal intent understanding.

## 2. Dependencies 

We use anaconda to create python environment:

```
conda create --name python=3.9
```

Install all required libraries:

```
pip install -r requirements.txt
```

## 3. Usage

The data can be downloaded through the following link:

```
https://cloud.tsinghua.edu.cn/d/58d20e316df24700ae8a/
```

The downloaded data contains two folders for each dataset, representing the ID data and OOD data. Taking MIntRec as an example, its directory structure is as follows:

```
-TMM_MIntOOD
  -MIntRec
    -video_data
      -swin_feats.pkl
    -audio_data
      -wavlm_feats.pkl
      -spectra_audio.pkl
    -train.tsv
    -dev.tsv
    -test.tsv

  -MIntRec-OOD
    -video_data
      -swin_feats.pkl
    -audio_data
      -wavlm_feats.pkl
      -spectra_audio.pkl
    -test.tsv
   
  ...
```

Notably, the `video_data` and `audio_data` in MELD-DA-OOD and IEMOCAP-OOD are the same as those in MELD-DA and IEMOCAP. Therefore, after decompression, the following command needs to be executed to copy `video_data` and `audio_data`:

```
cp -r MELD-DA/video_data MELD-DA-OOD/
cp -r MELD-DA/audio_data MELD-DA-OOD/
cp -r IEMOCAP/video_data IEMOCAP-OOD/
cp -r IEMOCAP/audio_data IEMOCAP-OOD/
```

You can evaluate the performance of our proposed MIntOOD under different settings by using the following commands and changing the parameters:

```
sh examples/run_train.sh
sh examples/run_test.sh

# Parameters in *.sh file
## Methods: mintood text mag_bert mult mmim tcl_map sdif spectra
## Dataset Configurations: MIntRec+MIntRec-OOD MELD-DA+MELD-DA-OOD IEMOCAP+IEMOCAP-DA-OOD
## OOD Detection methods: ma vim residual msp ma maxlogit
## Ablation Types: full text fusion_add fusion_concat sampler_beta wo_contrast wo_cosine wo_binary
## Note: If using SPECTRA, audio_feats and ood_audio_feats need to use features compatible with WavLM (replace audio_feats_path and ood_audio_feats_path with 'spectra_audio.pkl'). For details, refer to WavLM documentation at https://huggingface.co/docs/transformers/model_doc/wavlm.
```

You can change the hyper-parameters in the **configs** folder. The default hyper-parameters are the best on three datasets.

## Citations

If you are insterested in this work, and want to use the codes or results in this repository, please **star** this repository and **cite** the following works:
```
@article{zhang2024mintood,
      title={Multimodal Classification and Out-of-distribution Detection for Multimodal Intent Understanding}, 
      author={Hanlei Zhang and Qianrui Zhou and Hua Xu and Jianhua Su and Roberto Evans and Kai Gao},
      year={2024},
      journal={arXiv preprint arXiv:2412.12453},
}
```

