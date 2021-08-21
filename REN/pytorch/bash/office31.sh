#!/bin/sh


python train_image.py --net ResNet50 --dset office   --test_interval 500 --s_dset_path ../data/office/webcam_list.txt \
  --t_dset_path ../data/office/amazon_list.txt CDAN   --output_path w2a1031 --gpu_id 1
python train_image.py --net ResNet50 --dset office   --test_interval 500 --s_dset_path ../data/office/webcam_list.txt \
  --t_dset_path ../data/office/amazon_list.txt CDAN   --output_path w2a1032 --gpu_id 1
python train_image.py --net ResNet50 --dset office   --test_interval 500 --s_dset_path ../data/office/webcam_list.txt \
  --t_dset_path ../data/office/amazon_list.txt CDAN   --output_path w2a1033 --gpu_id 1
python train_image.py --net ResNet50 --dset office   --test_interval 500 --s_dset_path ../data/office/webcam_list.txt \
  --t_dset_path ../data/office/amazon_list.txt CDAN   --output_path w2a1034 --gpu_id 1
python train_image.py --net ResNet50 --dset office   --test_interval 500 --s_dset_path ../data/office/webcam_list.txt \
  --t_dset_path ../data/office/amazon_list.txt CDAN  --output_path w2a1035 --gpu_id 1
python train_image.py --net ResNet50 --dset office   --test_interval 500 --s_dset_path ../data/office/webcam_list.txt \
  --t_dset_path ../data/office/amazon_list.txt CDAN  --output_path w2a1036 --gpu_id 1
