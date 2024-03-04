import os
import pickle
# from tqdm import tqdm
from datetime import datetime

import numpy as np

import os
import pickle
import torch

def get_color_mnist():

    #Check if data folder exists
    cur_path = os.path.dirname(os.path.abspath(__file__))
    save_path=os.path.join(cur_path, 'data', 'color')
    train_path=os.path.join(save_path, 'train')
    test_path=os.path.join(save_path, 'valid')

    print("Loading data...")
    # data = np.load(os.path.join(save_path, filename))
    train_data_path = os.path.join(train_path, "images.npy")
    train_data = np.load(train_data_path)

    attr_train_data_path = os.path.join(train_path, "attrs.npy")
    attr_train_data = np.array(np.load(attr_train_data_path), dtype=object)[:, 1]

    print(train_data.shape, attr_train_data.shape)
    print(attr_train_data)

    test_data_path = os.path.join(test_path, "images.npy")
    test_data = np.load(test_data_path)

    attr_test_data_path = os.path.join(test_path, "attrs.npy")
    attr_test_data = np.array(np.load(attr_test_data_path), dtype=object)[:, 1]

    print(test_data.shape, attr_test_data.shape)
    print(attr_test_data)


    # colors_path = os.path.join(save_path, "colors.th")
    # mean_color = torch.load(colors_path)
    # attr_names_path = os.path.join(save_path, "attr_names.pkl")
    # with open(attr_names_path, "rb") as f:
    #     attr_names = pickle.load(f)

    # print(mean_color.shape)
    
    # print(mean_color, attr_names)
    # self.num_attrs =  self.attr.size(1)
    # self.set_query_attr_idx(query_attr_idx)
    # self.transform = transform

    return train_data/np.float32(255.), attr_train_data, test_data/np.float32(255.), attr_test_data

# class AttributeDataset(Dataset):
#     def __init__(self, root, split, query_attr_idx=None, transform=None):
#         super(AttributeDataset, self).__init__()
#         data_path = os.path.join(root, split, "images.npy")
#         self.data = np.load(data_path)
        
#         attr_path = os.path.join(root, split, "attrs.npy")
#         self.attr = torch.LongTensor(np.load(attr_path))

#         colors_path = os.path.join("./data", "resource", "colors.th")
#         mean_color = torch.load(colors_path)
#         attr_names_path = os.path.join(root, "attr_names.pkl")
#         with open(attr_names_path, "rb") as f:
#             self.attr_names = pickle.load(f)
        
#         self.num_attrs =  self.attr.size(1)
#         self.set_query_attr_idx(query_attr_idx)
#         self.transform = transform
    
#     def set_query_attr_idx(self, query_attr_idx):
#         if query_attr_idx is None:
#             query_attr_idx = torch.arange(self.num_attrs)
        
#         self.query_attr = self.attr[:, query_attr_idx]
        
#     def __len__(self):
#         return self.attr.size(0)

#     def __getitem__(self, index):
#         image, attr = self.data[index], self.query_attr[index]
#         if self.transform is not None:
#             image = self.transform(image)

#         return image, attr