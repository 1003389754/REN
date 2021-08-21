#!/bin/sh

python test_image.py --net ResNet50 --dset office  --s_dset_path ../data/office/webcam_list.txt   \\
  --t_dset_path ../data/office/amazon_list.txt --model_dir ./logs/con2/webcam2amazon_CDAN --gpu_id 0
