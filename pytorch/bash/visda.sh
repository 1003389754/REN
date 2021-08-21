#!/bin/sh

python train_image.py --net ResNet50 --dset visda --test_interval 500 --s_dset_path ../data/visda-2017/train_list.txt \
  --t_dset_path ../data/visda-2017/validation_list.txt CDAN --output_path visda1 --gpu_id 1
python train_image.py --net ResNet50 --dset visda --test_interval 500 --s_dset_path ../data/visda-2017/train_list.txt \
  --t_dset_path ../data/visda-2017/validation_list.txt CDAN --output_path visda2 --gpu_id 1
python train_image.py --net ResNet50 --dset visda --test_interval 500 --s_dset_path ../data/visda-2017/train_list.txt \
  --t_dset_path ../data/visda-2017/validation_list.txt CDAN --output_path visda3 --gpu_id 1