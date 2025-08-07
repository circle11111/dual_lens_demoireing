python train_gan.py \
 --dataset='mixed' \
 --data_path='/media/dongxuan-3090x2/74c55aac-55b2-438e-9ff8-b42e5c135289/usrs/sxy_tpami_moire_sintel/mixed_dataset' \
 --save_dir='checkpoints' \
 --epochs=100 \
 --train_batch_size=2 \
 --val_batch_size=2 \
 --threads=1 \
#  --load_dir='checkpoints' \ the following lines is uesd for finetuning
#  --G_filename='G_ssim.pth' \
#  --D_filename='D_ssim.pth' \
 