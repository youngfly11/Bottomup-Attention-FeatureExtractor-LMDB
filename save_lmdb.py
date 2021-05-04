# -*- encoding: utf-8 -*-
'''
@Time        :2021/04/27 09:50:45
@Author      :Yongfei Liu
@Email       :liuyf3@shanghaitech.edu.cn
'''

"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

convert image npz to LMDB
"""
import argparse
import glob
import io
import json
import multiprocessing as mp
import os
from os.path import basename, exists
from cytoolz import curry
import numpy as np
from tqdm import tqdm
import lmdb

import msgpack
import msgpack_numpy
msgpack_numpy.patch()

@curry
def load_npz(fname):
    try:
        dump = {}
        dump['features'] = np.load(fname)
    except Exception as e:
        # corrupted file
        print(f'corrupted file {fname}', e)
        dump = {}
        nbb = 0
    name = basename(fname).split('.')[0]
    return name, dump


def dumps_npz(dump, compress=False):
    with io.BytesIO() as writer:
        if compress:
            np.savez_compressed(writer, **dump, allow_pickle=True)
        else:
            np.savez(writer, **dump, allow_pickle=True)
        return writer.getvalue()


def dumps_msgpack(dump):
    return msgpack.dumps(dump, use_bin_type=True)


def main(opts):

    db_name="flickr30k_lmdb_test"
    env = lmdb.open(f'{opts.output}/{db_name}', map_size=1024**4)
    txn = env.begin(write=True)

    # split_files = open('/root/dataspace/Flickr30k/flickr30k_annos/train.txt', 'r')
    # with open('/root/dataspace/Flickr30k/flickr30k_annos/train.txt', 'r') as load_f:
    #     files = load_f.readlines()

    with open('/root/dataspace/Flickr30k/flickr30k_annos/val.txt', 'r') as load_f:
        files = load_f.readlines()
    
    with open('/root/dataspace/Flickr30k/flickr30k_annos/test.txt', 'r') as load_f:
        files.extend(load_f.readlines())
    
    files = [fl.split('\t')[0] for fl in files]
    files_empty = []
    for fl in files:
        if fl not in files_empty:
            files_empty.append(fl)
    
    files = [f"{opts.img_dir}/{fl}.npy" for fl in files_empty]
    print("num of files in val&test:", len(files))


    feat_files = glob.glob(f"{opts.img_dir}/*.npy")
    print(len(feat_files))
    for fl in files:
        if fl not in feat_files:
            print(fl)

    load = load_npz()
    name2nbb = {}
    with mp.Pool(opts.nproc) as pool, tqdm(total=len(files)) as pbar:
        for i, (fname, features) in enumerate(pool.imap_unordered(load, files, chunksize=128)):
            if features.shape[0]==3:
                print(fname)
                continue  # corrupted feature
            dump = dumps_msgpack(features)
            txn.put(key=fname.encode('utf-8'), value=dump)
            if i % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)
            name2nbb[fname] = 100
            pbar.update(1)
        txn.put(key=b'__keys__', value=json.dumps(list(name2nbb.keys())).encode('utf-8'))
        txn.commit()
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default=None, type=str, help="The input images.")
    parser.add_argument("--output", default=None, type=str, help="output lmdb")
    parser.add_argument('--nproc', type=int, default=4, help='number of cores used')
    args = parser.parse_args()
    main(args)

    # python save_lmdb.py --img_dir /root/dataspace/Flickr30k/flickr30k_feats --output /root/dataspace/Flickr30k
