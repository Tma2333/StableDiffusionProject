import pickle
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms 


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def parse_cifar_data (root_dir: str):
    root_dir = Path(root_dir)
    labels = []
    data = []
    for batch in root_dir.glob('data_batch_*'):
        raw_batch = unpickle(str(batch))
        labels.append(raw_batch[b'labels'])
        data.append(raw_batch[b'data'])
    train_labels = np.concatenate(labels)
    train_data = np.concatenate(data)
    train_data = train_data.reshape(len(train_data), 3, 32, 32).transpose(0,2,3,1)

    raw_batch = unpickle(str(root_dir/'test_batch'))
    test_labels = np.array(raw_batch[b'labels'])
    test_data = raw_batch[b'data']
    test_data = test_data.reshape(len(test_data), 3, 32, 32).transpose(0,2,3,1)

    meta_data = unpickle(str(root_dir/'batches.meta'))
    label_mapping = {class_name.decode('utf-8'): i for i, class_name in enumerate(meta_data[b'label_names'])}
    index_mapping = {i: class_name.decode('utf-8') for i, class_name in enumerate(meta_data[b'label_names'])}

    return train_data, train_labels, test_data, test_labels, label_mapping, index_mapping


class CIFAR10FullDataset(Dataset):
    MEAN = (0.49139968, 0.48215827, 0.44653124)
    STD = (0.24703233, 0.24348505, 0.26158768)
    def __init__(self, base_path: str = None, is_train: bool = True, standardize: bool = True ):
        if base_path is None:
            base_path = '/deep2/u/yma42/StableDiffusionProject/label_imbalance/data/cifar-10-batches-py'
        
        train_data, train_labels, test_data, test_labels, self.label_mapping, self.index_mapping = parse_cifar_data(base_path)

        if is_train:
            self.data = train_data
            self.labels = train_labels
        else:
            self.data = test_data
            self.labels = test_labels

        t = []
        
        t.append(transforms.ToTensor())
        if standardize:
            t.append(transforms.Normalize(mean=self.MEAN, std=self.STD))
        if is_train:
            t.append(transforms.RandomCrop(32, padding=4))
            t.append(transforms.RandomHorizontalFlip())
        

        self.preprocessing = transforms.Compose(t)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        x = self.preprocessing(x)
        return x, y


class CIFAR10ImbalanceDataset (CIFAR10FullDataset):
    def __init__(self, base_path: str = None, is_train: bool = True, standardize: bool = True, 
                       target_cls: int = 3, remove_ratio: float = 0.99):
        super().__init__(base_path, is_train, standardize)
        assert is_train, f'You should only use this dataset for trinaing'

        self.target_cls = target_cls
        # find all class idx
        cls_idx = np.where(self.labels==target_cls)[0]
        num_per_cls = len(cls_idx)
        num_to_remove = int(num_per_cls * remove_ratio) 
        # randomly pick sample to removec
        idx = np.random.choice(num_per_cls, num_to_remove, replace=False)
        idx_to_remove = cls_idx[idx] 
        subset_idx = np.ones(len(self.data), dtype=np.bool8)
        subset_idx[idx_to_remove] = False
        self.data = self.data[subset_idx]
        self.labels = self.labels[subset_idx]
        

class CIFAR10OverSampleDataset (CIFAR10ImbalanceDataset):
    def __init__(self, base_path: str = None, is_train: bool = True, standardize: bool = True, 
                       target_cls: int = 3, remove_ratio: float = 0.99):
        super().__init__(base_path, is_train, standardize, target_cls, remove_ratio)
        assert is_train, f'You should only use this dataset for trinaing'
        target_idx = self.labels == target_cls
        target_data = self.data[target_idx].copy()
        target_data_dup = target_data.repeat(99, 0)
        target_label_dup = np.array([target_cls]).repeat(4950)
        self.data = np.concatenate([self.data, target_data_dup], axis=0)
        self.labels = np.concatenate([self.labels, target_label_dup], axis=0)


class CIFAR10OverSampleRandomAugDataset (CIFAR10ImbalanceDataset):
    def __init__(self, base_path: str = None, is_train: bool = True, standardize: bool = True, 
                       target_cls: int = 3, remove_ratio: float = 0.99):
        super().__init__(base_path, is_train, standardize, target_cls, remove_ratio)
        assert is_train, f'You should only use this dataset for trinaing'
        target_idx = self.labels == target_cls
        target_data = self.data[target_idx].copy()
        target_data_dup = target_data.repeat(99, 0)
        target_label_dup = np.array([target_cls]).repeat(4950)

        t = []
        t.append(transforms.ToTensor())
        t.append(transforms.ColorJitter(brightness=.5, hue=.3))
        t.append(transforms.RandomPerspective(distortion_scale=0.2, p=1.0))
        t.append(transforms.RandomRotation(degrees=(0, 180)))
        t.append(transforms.RandomApply([transforms.RandomSolarize(threshold=0.75)], 0.5))
        t.append(transforms.ToPILImage())

        random_aug = transforms.Compose(t)
        print('process augmentation')
        for i, img in enumerate(target_data_dup):
            img = random_aug(img)
            img = np.array(img)
            target_data_dup[i] = img

        self.data = np.concatenate([self.data, target_data_dup], axis=0)
        self.labels = np.concatenate([self.labels, target_label_dup], axis=0)


class CIFAR10StableDiffusionDataset (CIFAR10ImbalanceDataset):
    def __init__(self, base_path: str = None, sb_path: str = None, is_train: bool = True, standardize: bool = True, 
                       target_cls: int = 3, remove_ratio: float = 0.99):
        super().__init__(base_path, is_train, standardize, target_cls, remove_ratio)
        assert is_train, f'You should only use this dataset for trinaing'
        assert target_cls==3, f'Stable Diffusion only generate for cat image which is class 3'

        if sb_path is None:
            sb_path = '/deep2/u/yma42/StableDiffusionProject/label_imbalance/data/sb_randm_sample.npy'
        sb_sample = np.load(sb_path)
        target_label_sb = np.array([target_cls]).repeat(len(sb_sample))

        self.data = np.concatenate([self.data, sb_sample], axis=0)
        self.labels = np.concatenate([self.labels, target_label_sb], axis=0)
