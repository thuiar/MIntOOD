## Methods: mintood text mag_bert mult mmim tcl_map sdif spectra
## Dataset Configurations: MIntRec+MIntRec-OOD MELD-DA+MELD-DA-OOD IEMOCAP+IEMOCAP-DA-OOD
## OOD Detection methods: ma vim residual msp ma maxlogit
## Ablation Types: full text fusion_add fusion_concat sampler_beta wo_contrast wo_cosine wo_binary
## Note: If using SPECTRA, audio_feats and ood_audio_feats need to use features compatible with WavLM (replace audio_feats_path and ood_audio_feats_path with 'spectra_audio.pkl'). For details, refer to WavLM documentation at https://huggingface.co/docs/transformers/model_doc/wavlm.


for method in 'sdif'
do
    for text_backbone in 'bert-base-uncased' 
    do
        for ood_dataset in  'MIntRec-OOD' 
        do
            for dataset in 'MIntRec'
            do
                for ood_detection_method in 'ma'
                do
                    for ablation_type in 'full' 
                    do
                        python run.py \
                        --dataset $dataset \
                        --data_path '/home/sharing/Datasets' \
                        --ood_dataset $ood_dataset \
                        --logger_name ${method}_${ood_detection_method} \
                        --multimodal_method $method \
                        --method ${method}\
                        --ood_detection_method $ood_detection_method \
                        --ablation_type $ablation_type \
                        --train \
                        --ood \
                        --tune \
                        --save_results \
                        --save_model \
                        --gpu_id '0' \
                        --video_feats_path 'swin_feats.pkl' \
                        --audio_feats_path 'wavlm_feats.pkl' \
                        --ood_video_feats_path 'swin_feats.pkl' \
                        --ood_audio_feats_path 'wavlm_feats.pkl' \
                        --text_backbone $text_backbone \
                        --config_file_name ${method}_${dataset} \
                        --output_path "outputs" \
                        --results_file_name 'results_mintood_train.csv'
                    done
                done 
            done
        done
    done
done
