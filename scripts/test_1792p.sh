############## To test images at 1792 x 1792 #############

# Adding instances and encoded features
python /pix2pixHD/test.py --name ha2hel --which_epoch latest --netG global --checkpoints_dir /checkpoints/  --input_nc 1 --output_nc 1 --ngf 64 --resize_or_crop none --label_nc 0 --no_instance --no_flip --dataroot /datasets
