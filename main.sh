python main.py \
    --seed=2022 \
    --experiment_name="EfficientNetV1" \
    --root_dir="DATASET" \
    --default_root_dir="experiment" \
    --dataset="CIFAR10" \
    --num_classes=10 \
    --transforms="BASE" \
    --image_channels=3 \
    --image_size=224 \
    --batch_size=32 \
    --model="EfficientNetV1"  \
    --model_type="b0" \
    --dropout_rate=0.5 \
    --lr=0.03 \
    --momentum=0.0 \
    --weight_decay=0.0 \
    --callbacks_monitor="val/acc" \
    --callbacks_mode="max" \
    --earlystooping_min_delta=0.1 \
    --earlystooping_patience=6 \
    --log_every_n_steps=10 \
    --gpus=1 \
    --max_epochs=1 \
    --detect_anomaly=True 




