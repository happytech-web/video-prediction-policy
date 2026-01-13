accelerate launch step2_train_action_calvin.py \
        --root_data_dir /home/humanoid/linglong/vpp/video-prediction-policy/calvin/dataset/calvin_debug_dataset \
        --video_model_path /home/humanoid/linglong/vpp/video-prediction-policy/svd-calvin-ckpt/svd-robot-calvin-ft \
        --text_encoder_path /home/humanoid/linglong/vpp/video-prediction-policy/svd-calvin-ckpt/clip-vit-base-patch32 \
        --lang_folder lang_annotations \
        --task_index_json ./calvin_task_index_debug.json \
        --use_kmeans \
        --kmeans_k 5 \
        --kmeans_refresh_interval 2 \
        --lambda_contra 0.1 --lambda_proto 0.05 --lambda_metric 0.05
