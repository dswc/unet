from conf import *
from model import *
from data import *


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train():
    data_gen_args = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest')
    myGene = trainGenerator(batch_size, train_img_path, train_img_dir, train_label_dir, data_gen_args, save_to_dir = None)
    model_checkpoint = ModelCheckpoint(model_name, monitor='loss', verbose=1, save_best_only=True)
    model = unet()
    model.fit_generator(myGene, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[model_checkpoint])

if __name__ == '__main__':
    train()
    print('[Info] Done.')
