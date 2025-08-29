'''
Description: 
Author: Rigel Ma
Date: 2024-04-22 13:40:22
LastEditors: Rigel Ma
LastEditTime: 2024-04-22 22:13:15
FilePath: download.py
'''

import requests
from bs4 import BeautifulSoup
import os
from fuzzywuzzy import fuzz
from tqdm import tqdm
import math
import shutil
import zipfile
import gzip


def download(url_root, product_name=""):
    r = requests.get(url_root)
    if r.status_code != 200:
        raise "connect error!, return {}.".format(r.status_code)

    soup = BeautifulSoup(r.content, 'lxml')
    products = soup.select("body table tr td a")

    candidate_product_names = []
    candidate_product_links = []

    for product in products:
        link_ = product['href']
        name_ = product.text
        
        if '/' not in link_:
            candidate_product_links.append(link_)
            candidate_product_names.append(name_)
        
    if product_name != "":
        print("matching!")
        match_names = []
        match_links = []
        for idx, candidate_name in enumerate(candidate_product_names):
            if fuzz.ratio(product_name, candidate_name) > 20:
                print(product_name, candidate_name, fuzz.ratio(product_name, candidate_name))
                match_names.append(candidate_name)
                match_links.append(candidate_product_links[idx])
        if len(match_names) == 0:
            print(f'can not find the target file "{product_name}". Files can be downloaded are follows:')
            match_names = candidate_product_names
            match_links = candidate_product_links
    else:
        match_names = candidate_product_names
        match_links = candidate_product_links
    

    for idx in range(len(match_names)):
        print(f"{idx}. {match_names[idx]}")

    download_ids = input('please input the ids of files you want to download\nsplit by , if many files:')
    download_ids = download_ids.replace(' ','').split(',')

    # check validation of input
    for id in download_ids:
        try:
            _ = int(id)
        except ValueError:
            raise "invalid id!"


    for id in download_ids:
        product_name_ = match_names[int(id)]
        product_link_ = match_links[int(id)]

        if url_root[-1] == '/':
            url_ = url_root + product_link_
        else:
            url_ = url_root + '/' + product_link_
        product_name_without_suffix = product_name_.split('.')[0]
        print(f'start download {match_names[int(id)]}')
        download_file(url_, f'./dataset/{product_name_without_suffix}/{product_name_}')



def download_file(url, target_path):
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)

    r = requests.get(url, stream=True)

    if r.status_code == 200:
        total_size = int(r.headers.get('content-length', 0))

        if os.path.exists(target_path):
            exsit_size = os.stat(target_path).st_size
        else:
            exsit_size = -1

        if exsit_size != total_size:

            # # write directly
            # with open(dst_file, 'wb') as f:
            #     f.write(r.content)
            #     f.close()

            # stream
            block_size = 1024   # 1024 bytes as a block
            total_size = int(r.headers.get('content-length', 0)) # total size
            num_iters = math.ceil(total_size / block_size)
            with open(target_path, 'wb') as f:
                for data in tqdm(r.iter_content(block_size),total=num_iters, unit="KB", unit_scale=True):  # each block is 1024 bytes, i.e. 1KB 
                    f.write(data)

        else:
            print('File exsits!')

        filename = target_path.split('/')[-1]
        if filename.endswith("zip"):
            extract_zip(target_path)
    
    else:
        print('Fail to donwload')
        r.raise_for_status()


def extract_zip(target_path):
    # extract
    file_name = os.path.basename(target_path).split('.')[0]
    target_dir = '/'.join(target_path.split('/')[:-1])

    # target_path: xxx/xxx.zip -> xxx/
    print('Unzipping')
    zip_file = zipfile.ZipFile(target_path)
    zip_file.extractall(target_dir)

    
def extract_gzip(target_path):
    file_name = os.path.basename(target_path).split('.')[0]
    target_dir = '/'.join(target_path.split('/')[:-1])

    # target_path: xxx/xxx.gz -> xxx/xxx
    with gzip.open(target_path, 'rb') as zf, open(os.path.join(target_dir, file_name), 'wb') as f:
        shutil.copyfileobj(zf, f)

if __name__ == "__main__":
    download("https://files.grouplens.org/datasets/movielens/")

