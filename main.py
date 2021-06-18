from model import *
from data import *
import os
import cv2
import skimage.transform as trans

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

is_train = False
epochs = 200
steps_per_epoch = 210
batch_size = 10
crop_height = 256
crop_width = 256
train_img_path = r'data/matek/train'
test_img_path = r'data/matek/test'
save_img_path = r'data/matek/result'
model_name = 'unet_matek.hdf5'

if is_train:
    data_gen_args = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest')
    myGene = trainGenerator(batch_size, train_img_path, 'crop_img', 'crop_label', data_gen_args, save_to_dir = None)
    model_checkpoint = ModelCheckpoint(model_name, monitor='loss', verbose=1, save_best_only=True)
    model = unet()
    model.fit_generator(myGene, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[model_checkpoint])
else:
    model = unet(model_name)

# testGene = testGenerator(test_img_path)
# num_test_imgs = len(os.listdir(test_img_path))
# results = model.predict_generator(testGene, num_test_imgs, verbose=1)
# saveResult(save_img_path, results)

image_names = os.listdir(test_img_path)
for i in image_names:
    img = cv2.imread(os.path.join(test_img_path, i))
    
    if img is None:
        print(i)
        continue
    
    b, g, r = cv2.split(img)
    img = g.astype(np.float32) / 255
    height, width = img.shape
    num_crop_height = int(np.ceil(height / crop_height))
    num_crop_width = int(np.ceil(width / crop_width))
    
    dst_img = np.zeros((height, width), dtype=np.float32)
    for row in range(num_crop_height):
        row_s = row * crop_height
        row_e = row * crop_height + crop_height
        if row_e > height:
            row_s = height - crop_height
            row_e = height
            
        for col in range(num_crop_width):
            col_s = col * crop_width
            col_e = col * crop_width + crop_width
            if col_e > width:
                col_s = width - crop_width
                col_e = width
            
            croped_img = img[row_s:row_e, col_s:col_e]
            
            # croped_img = trans.resize(croped_img, (256, 256))
            croped_img = np.reshape(croped_img, croped_img.shape+(1,))
            croped_img = np.reshape(croped_img, (1,)+croped_img.shape)
            
            # input_1 to have shape (None, 256, 256, 1)
            res = model.predict(croped_img)
            
            # io.imsave(os.path.join(save_img_path, '%d_%d.bmp' % (row, col)), res[0, :, :, 0])
            dst_img[row_s:row_e, col_s:col_e] = res[0, :, :, 0]
    dst_img = dst_img * 255
    dst_img = dst_img.astype(np.uint8)
    dst_img[dst_img > 127] = 255
    dst_img[dst_img <= 127] = 0
    cv2.imwrite(os.path.join(save_img_path, i), dst_img)
