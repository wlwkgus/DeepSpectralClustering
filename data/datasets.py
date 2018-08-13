import os
import scipy.io as sio
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
from PIL import ImageFile
from utils import utils
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CarsDataset(Dataset):
    def __init__(self, root_dir=None, train=None, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.train_count = 8054
        self.test_count = 8131
        if self.train:
            self.offset = 0
        else:
            self.offset = self.train_count
        self.image_dir = os.path.join(self.root_dir, 'car_ims_cropped')
        self.index_table = sio.loadmat(
            os.path.join(self.root_dir, 'cars_annos.mat')
        )['annotations'][0]
        # label table : list of ( label, directory ).
        self.label_table = [(row[5][0][0], row[0][0]) for row in self.index_table]
        # call class by self.classes[index][0]
        self.classes = sio.loadmat(os.path.join(self.root_dir, 'devkit/cars_meta.mat'))['class_names'][0]
        if not os.path.exists(self.image_dir):
            self.crop_images()

    def crop_images(self):
        # To generate crop car images.
        image_dir = os.path.join(self.root_dir, 'car_ims_cropped')
        print("Cropping Images...")
        utils.mkdir(image_dir)
        for row in tqdm(self.index_table):
            name = row[0][0]

            image = Image.open(
                os.path.join(self.root_dir, name)
            )
            image = image.crop(
                (
                    row[1][0][0],
                    row[2][0][0],
                    row[3][0][0],
                    row[4][0][0],
                )
            )
            image.save(os.path.join(image_dir, name.split('/')[-1]))

    def __len__(self):
        if self.train:
            return self.train_count
        else:
            return self.test_count

    def __getitem__(self, idx):
        if not self.train:
            idx += self.train_count
        sample = dict()
        sample['img'] = Image.open(os.path.join(self.image_dir, self.label_table[idx][1].split('/')[-1]))
        sample['label'] = self.label_table[idx][0]
        if self.transform is not None:
            sample['img'] = self.transform(sample['img'])
        return sample


class CustomDataset(Dataset):
    # This is an example of how to generate custom data set.
    def __init__(self, transform=None):
        self.transform = transform
        self.image_dir = '/images'
        self.index_table = pd.read_csv('./clusters.csv')

    def __len__(self):
        return len(self.index_table)

    def __getitem__(self, idx):
        sample = dict()
        card_id = self.index_table.loc[idx].card_id
        cluster_id = self.index_table.loc[idx].cluster
        sample['img'] = Image.open(os.path.join(self.image_dir, '{}.jpg'.format(card_id)))
        sample['label'] = cluster_id
        if self.transform is not None:
            sample['img'] = self.transform(sample['img'])
        return sample
