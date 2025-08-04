############## To train images at 1792 x 1792 #############

# Adding instances and encoded features
python /pix2pixHD/train.py --name ha2hel --checkpoints_dir /checkpoints/ --loadSize 1792 --save_epoch_freq 10 --niter 500 --niter_decay 500 --input_nc 1 --output_nc 1 --ngf 64 --resize_or_crop none --label_nc 0 --no_instance --no_flip --dataroot /datasets/ha2hel/
