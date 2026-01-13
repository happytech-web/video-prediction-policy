accelerate launch step2_train_action_calvin.py \
        --root_data_dir /data01/linglong/datasets/task_ABC_D \
        --video_model_path /home/humanoid/linglong/vpp/video-prediction-policy/svd-calvin-ckpt/svd-robot-calvin-ft \
        --text_encoder_path /home/humanoid/linglong/vpp/video-prediction-policy/svd-calvin-ckpt/clip-vit-base-patch32 \
        --lang_folder lang_clip_resnet50 \
        --task_index_json ./calvin_task_index_abc_d.json \
        --use_kmeans \
        --kmeans_k 30 \
        --kmeans_refresh_interval 1 \
        --lambda_contra 0.1 --lambda_proto 0.1 --lambda_metric 0.1 \
        --use_wandb --wandb_project calvin_abcd \
        --enable_grouped_sampling


#   --lang_folder lang_paraphrase-MiniLM-L3-v2
#   --min_per_skill <int> (default is 2)
#   --max_skills_per_batch <int> (default is batch_size // min_per_skill)
#   --drop_last_grouped: (default is false)
