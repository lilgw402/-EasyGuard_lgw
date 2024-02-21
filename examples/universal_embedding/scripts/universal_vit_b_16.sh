# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 --master_port 43558 \
path/to/EasyGuard/tools/TORCHRUN \
main.py \
            --batch_size           128 \
            --dataset              product_all \
            --debug                0 \
            --epochs               100 \
            --lr                   5e-05 \
            --lr_pfc_weight        10.0 \
            --input_size           224 \
            --gradient_acc         1 \
            --model_name           ViT-B/16 \
            --margin_loss_m1       1.0 \
            --margin_loss_m2       0.25 \
            --margin_loss_m3       0.0 \
            --margin_loss_s        32.0 \
            --margin_loss_filter   0.0 \
            --num_workers          8 \
            --num_feat             768 \
            --optimizer            adamw \
            --output_dim           768 \
            --output               /tmp/tmp_for_training \
            --output_path          ./checkpoints_universal_v3
            --resume               NULL \
            --sample_rate          1.0 \
            --seed                 1024 \
            --weight_decay         0 
