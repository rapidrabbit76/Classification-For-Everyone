python main.py \
    --seed=2022 \
    --experiment_name="VGG" \
    --root_dir="DATASET" \
    --default_root_dir="experiment" \
    --dataset="CIFAR10" \
    --num_classes=10 \
    --transforms="BASE" \
    --image_channels=3 \
    --image_size=224 \
    --batch_size=256 \
    --model="VGG"  \
    --model_type="VGG13" \
    --dropout_rate=0.5 \
    --lr=0.01 \
    --momentum=0.9 \
    --weight_decay=0.0005 \
    --callbacks_monitor="val/acc" \
    --callbacks_mode="max" \
    --earlystooping_min_delta=0.1 \
    --earlystooping_patience=30 \
    --log_every_n_steps=10 \
    --gpus=1 \
    --max_epochs=100 \
    --detect_anomaly=True 




