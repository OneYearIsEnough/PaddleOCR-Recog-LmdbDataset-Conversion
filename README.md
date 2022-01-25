# Lmdb Dataset Format Conversion Tool

In [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)'s text recognition tasks, it supports two dataset format, one is `SimpleDataset`(or Icdar2015 dataset format), and another is `LmdbDataset`。

When your training datasets is very large, on account of lots of memory reads, which result in a low system io speed, so at this time using `SimpleDataset`(or Icdar2015 dataset format) may slow down your training speed extremely!

To solve this problem,  I wrote a lmdb dataset format conversion python script which can transform `SimpleDataset` fomat to `LmdbDataset` format.

# Python package dependencies

- lmdb
- opencv-python
- numpy

You can also use `pip install -r requirements.txt` to install those packages.

# How to use

There is only one executable python script file named `make_lmdb.py` in this project, so it's very easy to use.

Parameters usage description:

```shell
Args:
    --data_root_dir: A dir which contains total imgs or total img subdirs(e.g. train_data).
    --label_file_paths: Txt files to store the image path which based on ${data_root_dir} 
                        param and label. If you have more than one txt files, please use 
                        space char to split them(e.g. label1.txt label2.txt label3.txt).
    --delimiter: Only support 'blank' and 'tab' delimiter in ${label_file_paths} to split
                 image path and image label. By default, the image path and image label are 
                 split with 'tab', which is \t. I also recommend you to use \t as delimiter.
    --lmdb_out_dir: Output lmdb dir.
    --check: If declared, it will check every image whether it is valid or not and throw 
             invalid images away, thus we can get a cleaner lmdb dataset, but this could 
             result in inefficiencies.
```

# Demo

For example, if the training set has the following file structure:

```shell
|-train_data
  |-rec
    |- gt_label1.txt
    |- gt_label2.txt
    |- gt_label3.txt
    |- train
        |- word_001.png
        |- word_002.jpg
        |- word_003.jpg
        | ...
```

We can use the following code to generate lmdb dataset:

```shell
python3 make_lmdb.py \
    --data_root_dir train_data,
    --label_file_paths train_data/rec/gt_label1.txt train_data/rec/gt_label2.txt train_data/rec/gt_label3.txt \
    --delimiter tab \
    --lmdb_out_dir ${output lmdb dir you specified} \
```

Commonly, in text recognition task, the training data is very large and we can't tolerate a low conversion speed, so it is better not to declare the parameter`check`.