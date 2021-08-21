#!/bin/sh

python train_image.py --net ResNet50 --dset image-clef --test_interval 500 --s_dset_path ../data/image-clef/c_list.txt \
  --t_dset_path ../data/image-clef/i_list.txt CDAN   --output_path image-clef1 --gpu_id 1
python train_image.py --net ResNet50 --dset image-clef --test_interval 500 --s_dset_path ../data/image-clef/c_list.txt \
  --t_dset_path ../data/image-clef/i_list.txt CDAN   --output_path image-clef2 --gpu_id 1
python train_image.py --net ResNet50 --dset image-clef --test_interval 500 --s_dset_path ../data/image-clef/c_list.txt \
  --t_dset_path ../data/image-clef/i_list.txt CDAN   --output_path image-clef3 --gpu_id 1


python train_image.py --net ResNet50 --dset image-clef --test_interval 500 --s_dset_path ../data/image-clef/i_list.txt \
  --t_dset_path ../data/image-clef/c_list.txt CDAN   --output_path image-clef1 --gpu_id 1
python train_image.py --net ResNet50 --dset image-clef --test_interval 500 --s_dset_path ../data/image-clef/i_list.txt \
  --t_dset_path ../data/image-clef/c_list.txt CDAN   --output_path image-clef2 --gpu_id 1
python train_image.py --net ResNet50 --dset image-clef --test_interval 500 --s_dset_path ../data/image-clef/i_list.txt \
  --t_dset_path ../data/image-clef/c_list.txt CDAN   --output_path image-clef3 --gpu_id 1

python train_image.py --net ResNet50 --dset image-clef --test_interval 500 --s_dset_path ../data/image-clef/i_list.txt \
  --t_dset_path ../data/image-clef/p_list.txt CDAN   --output_path image-clef1 --gpu_id 1
python train_image.py --net ResNet50 --dset image-clef --test_interval 500 --s_dset_path ../data/image-clef/i_list.txt \
  --t_dset_path ../data/image-clef/p_list.txt CDAN   --output_path image-clef2 --gpu_id 1
python train_image.py --net ResNet50 --dset image-clef --test_interval 500 --s_dset_path ../data/image-clef/i_list.txt \
  --t_dset_path ../data/image-clef/p_list.txt CDAN   --output_path image-clef3 --gpu_id 1

python train_image.py --net ResNet50 --dset image-clef --test_interval 500 --s_dset_path ../data/image-clef/p_list.txt \
  --t_dset_path ../data/image-clef/c_list.txt CDAN   --output_path image-clef1 --gpu_id 1
python train_image.py --net ResNet50 --dset image-clef --test_interval 500 --s_dset_path ../data/image-clef/p_list.txt \
  --t_dset_path ../data/image-clef/c_list.txt CDAN   --output_path image-clef2 --gpu_id 1
python train_image.py --net ResNet50 --dset image-clef --test_interval 500 --s_dset_path ../data/image-clef/p_list.txt \
  --t_dset_path ../data/image-clef/c_list.txt CDAN   --output_path image-clef3 --gpu_id 1


python train_image.py --net ResNet50 --dset image-clef --test_interval 500 --s_dset_path ../data/image-clef/p_list.txt \
  --t_dset_path ../data/image-clef/i_list.txt CDAN   --output_path image-clef1 --gpu_id 1
python train_image.py --net ResNet50 --dset image-clef --test_interval 500 --s_dset_path ../data/image-clef/p_list.txt \
  --t_dset_path ../data/image-clef/i_list.txt CDAN   --output_path image-clef2 --gpu_id 1
python train_image.py --net ResNet50 --dset image-clef --test_interval 500 --s_dset_path ../data/image-clef/p_list.txt \
  --t_dset_path ../data/image-clef/i_list.txt CDAN   --output_path image-clef3 --gpu_id 1