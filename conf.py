import os
import math
import argparse


parser = argparse.ArgumentParser(description='Training')
parser.add_argument("--dataset", help="The path of your data")
parser.add_argument("--epochs", help="Choose epochs you want each trial be trained.", type=int, default=0)
args = parser.parse_args()
print(args.dataset)
print(args.epochs)
epochs = args.epochs

root_path = os.path.join('data', args.dataset)
train_img_path = os.path.join(root_path, r'train')
train_original_img_dir = 'image'
train_original_label_dir = 'label'
train_img_dir = 'crop_img'
train_label_dir = 'crop_label'
test_img_path = os.path.join(root_path, r'test/image')
save_img_path = os.path.join(root_path, r'test/result')
model_name = 'unet_matek.hdf5'

train_img_number = len(os.listdir(os.path.join(train_img_path, train_img_dir)))

# epochs = 30
batch_size = 10
steps_per_epoch = math.ceil(train_img_number / batch_size)
crop_height = 256
crop_width = 256
