# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 17:40:57 2020

@author: iant

READ AND WRITE HDF5 FILES TO/FROM DICTIONARIES
"""


# hdf5_filename = "20190102_190343_1p0a_SO_A_E_129"
# hdf5_filename_new = "20190102_190343_1p0a_SO_A_E_129_test"


def read_hdf5_to_dict(hdf5_filename):

    import h5py

    with h5py.File(hdf5_filename+".h5",'r') as f:
    
        datasets = {}
        for node, dataset in f.items():
            path = node
        #    print(path)
            if isinstance(f[path], h5py.Dataset):
                datasets[path] = dataset[...]
            else:
                for node2, dataset2 in f[node].items():
                    path = node+"/"+node2
        #            print(path)
                    if isinstance(f[path], h5py.Dataset):
                        datasets[path] = dataset2[...]
                    else:
                        for node3, dataset3 in f[node][node2].items():
                            path = node+"/"+node2+"/"+node3
        #                    print(path)
                            if isinstance(f[path], h5py.Dataset):
                                datasets[path] = dataset3[...]
                                
        attributes = {}
        for name in f.attrs:
            attributes[name] = f.attrs[name]

    return datasets, attributes            




def write_hdf5_from_dict(hdf5_filename_new, dataset_dict, attribute_dict, replace_datasets, replace_attributes, resize_len=0, resize_index=0):

    import h5py
    import numpy as np

    with h5py.File(hdf5_filename_new+".h5", "w") as f:
        for name, attr in attribute_dict.items():
            if name in replace_attributes.keys():
                f.attrs[name] = replace_attributes[name]
            else:
                f.attrs[name] = attr
        
        for node, dataset in dataset_dict.items():
            dtype = dataset.dtype
            if dataset.ndim == 1:
                if resize_len > 0:
                    if dataset.shape[0] == resize_len:
                        dataset = np.array(dataset[resize_index], dtype=dtype)
            if dataset.ndim == 2:
                if resize_len > 0:
                    if dataset.shape[0] == resize_len:
                        dataset = np.array(dataset[resize_index, :], dtype=dtype)
            
#            if node in replace_datasets.keys():
#                f[node] = np.array(replace_datasets[node], dtype=dtype)
#            else:
#                f[node] = dataset
            if node in replace_datasets.keys():
                data = np.array(replace_datasets[node], dtype=dtype)
            else:
                data = dataset

            if data.size>1:
                compression="gzip"
                shuffle=True
            else:
                compression=None
                shuffle=False

#            print(node, data.shape, data, data.size)
            f.create_dataset(node, data=data, dtype=dtype, compression=compression, shuffle=shuffle)
                