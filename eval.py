from conf import *
from model import *
from data import *
import os
import cv2
import numpy as np
# import skimage.transform as trans


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

            if row_s < 0:
                row_s = 0
            if col_s <0:
                col_s = 0
            cropped_img = img[row_s:row_e, col_s:col_e]
            cropped_h, cropped_w = cropped_img.shape
            if cropped_h < crop_height:
                cropped_img = np.pad(cropped_img, ((0, crop_height-cropped_h), (0, 0)), 'constant')
            if cropped_w < crop_width:
                cropped_img = np.pad(cropped_img, ((0, 0), (0, crop_width-cropped_w)), 'constant')

            # croped_img = trans.resize(croped_img, (256, 256))
            cropped_img = np.reshape(cropped_img, cropped_img.shape + (1,))
            cropped_img = np.reshape(cropped_img, (1,) + cropped_img.shape)

            # input_1 to have shape (None, 256, 256, 1)
            res = model.predict(cropped_img)

            # io.imsave(os.path.join(save_img_path, '%d_%d.bmp' % (row, col)), res[0, :, :, 0])
            dst_img[row_s:row_e, col_s:col_e] = res[0, 0:cropped_h, 0:cropped_w, 0]
    dst_img = dst_img * 255
    dst_img = dst_img.astype(np.uint8)
    dst_img[dst_img > 127] = 255
    dst_img[dst_img <= 127] = 0
    cv2.imwrite(os.path.join(save_img_path, i), dst_img)

print('[Info] Done.')
