#!/bin/sh

#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Art.txt \
#  --t_dset_path ../data/office-home/Clipart.txt CDAN   --output_path office_home1 --gpu_id 0
#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Art.txt \
#  --t_dset_path ../data/office-home/Clipart.txt CDAN   --output_path office_home2 --gpu_id 0
#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Art.txt \
#  --t_dset_path ../data/office-home/Clipart.txt CDAN   --output_path office_home3 --gpu_id 0
#
#
#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Art.txt \
#  --t_dset_path ../data/office-home/Product.txt CDAN   --output_path office_home1 --gpu_id 0
#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Art.txt \
#  --t_dset_path ../data/office-home/Product.txt CDAN   --output_path office_home2 --gpu_id 0
#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Art.txt \
#  --t_dset_path ../data/office-home/Product.txt CDAN   --output_path office_home3 --gpu_id 0
#
#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Art.txt \
#  --t_dset_path ../data/office-home/Real_World.txt CDAN   --output_path office_home1 --gpu_id 0
#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Art.txt \
#  --t_dset_path ../data/office-home/Real_World.txt CDAN   --output_path office_home2 --gpu_id 0
#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Art.txt \
#  --t_dset_path ../data/office-home/Real_World.txt CDAN   --output_path office_home3 --gpu_id 0
#
#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Clipart.txt \
#  --t_dset_path ../data/office-home/Real_World.txt CDAN   --output_path office_home1 --gpu_id 0
#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Clipart.txt \
#  --t_dset_path ../data/office-home/Real_World.txt CDAN   --output_path office_home2 --gpu_id 0
#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Clipart.txt \
#  --t_dset_path ../data/office-home/Real_World.txt CDAN   --output_path office_home3 --gpu_id 0
#
#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Clipart.txt \
#  --t_dset_path ../data/office-home/Product.txt CDAN   --output_path office_home1 --gpu_id 0
#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Clipart.txt \
#  --t_dset_path ../data/office-home/Product.txt CDAN   --output_path office_home2 --gpu_id 0
#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Clipart.txt \
#  --t_dset_path ../data/office-home/Product.txt CDAN   --output_path office_home3 --gpu_id 0
#
#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Clipart.txt \
#  --t_dset_path ../data/office-home/Art.txt CDAN   --output_path office_home1 --gpu_id 0
#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Clipart.txt \
#  --t_dset_path ../data/office-home/Art.txt CDAN   --output_path office_home2 --gpu_id 0
#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Clipart.txt \
#  --t_dset_path ../data/office-home/Art.txt CDAN   --output_path office_home3 --gpu_id 0

#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Art.txt \
#  --t_dset_path ../data/office-home/Real_World.txt CDAN   --output_path 1 --gpu_id 0
#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Art.txt \
#  --t_dset_path ../data/office-home/Real_World.txt CDAN   --output_path 2 --gpu_id 0
python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Art.txt \
  --t_dset_path ../data/office-home/Real_World.txt CDAN   --output_path 3 --gpu_id 0

python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Clipart.txt \
  --t_dset_path ../data/office-home/Real_World.txt CDAN   --output_path 1 --gpu_id 0
python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Clipart.txt \
  --t_dset_path ../data/office-home/Real_World.txt CDAN   --output_path 2 --gpu_id 0
python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Clipart.txt \
  --t_dset_path ../data/office-home/Real_World.txt CDAN   --output_path 3 --gpu_id 0

python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Real_World.txt \
  --t_dset_path ../data/office-home/Art.txt CDAN   --output_path 1 --gpu_id 0
python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Real_World.txt \
  --t_dset_path ../data/office-home/Art.txt CDAN   --output_path 2 --gpu_id 0
python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Real_World.txt \
  --t_dset_path ../data/office-home/Art.txt CDAN   --output_path 3 --gpu_id 0

#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Real_World.txt \
#  --t_dset_path ../data/office-home/Clipart.txt CDAN   --output_path office_home1 --gpu_id 0
#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Real_World.txt \
#  --t_dset_path ../data/office-home/Clipart.txt CDAN   --output_path office_home2 --gpu_id 0
#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Real_World.txt \
#  --t_dset_path ../data/office-home/Clipart.txt CDAN   --output_path office_home3 --gpu_id 0
#
#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Real_World.txt \
#  --t_dset_path ../data/office-home/Art.txt CDAN   --output_path office_home1 --gpu_id 0
#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Real_World.txt \
#  --t_dset_path ../data/office-home/Art.txt CDAN   --output_path office_home2 --gpu_id 0
#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Real_World.txt \
#  --t_dset_path ../data/office-home/Art.txt CDAN   --output_path office_home3 --gpu_id 0
#
#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Real_World.txt \
#  --t_dset_path ../data/office-home/Product.txt CDAN   --output_path office_home1 --gpu_id 0
#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Real_World.txt \
#  --t_dset_path ../data/office-home/Product.txt CDAN   --output_path office_home2 --gpu_id 0
#python train_image_MT_3.py --net ResNet50 --dset office-home --test_interval 500 --s_dset_path ../data/office-home/Real_World.txt \
#  --t_dset_path ../data/office-home/Product.txt CDAN   --output_path office_home3 --gpu_id 0
