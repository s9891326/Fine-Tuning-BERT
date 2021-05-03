import click
import numpy as np
import pandas as pd

from typing import List
from pathlib import Path
from absl import logging

import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from definitions import OLD_FOLDER, NEW_FOLDER

RANDOM_STATE = 43
NAMES = ["content", "label"]
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

# @click.group(context_settings=CONTEXT_SETTINGS)
# def cli():
#     pass


@click.command()
@click.option('--old_dataset_name', default="train.tsv", show_default=True,
              help='舊資料集名稱', type=str)
@click.option('--new_dataset_name', default="new_train.tsv", show_default=True,
              help='新資料集名稱', type=str)
@click.option('--output_dataset_name', default="train.tsv", show_default=True,
              help='混合資料集名稱', type=str)
@click.option('--mix_number', default=0, show_default=True,
              prompt=True, help="混合多少數量的訓練集，如果沒設定，"
                                "則是依照新資料集內的各標籤的數量進行舊資料的混合",
              type=int)
@click.option('--minimum_content_length', default=20, show_default=True,
              help="混合資料時，舊資料抓取時的最少字數限制", type=int)
@click.option('--maximum_digits', default=3, show_default=True,
              help="最大能接受資料筆數的位元個數，如果超過抓最小公倍數，沒超過則以10^maximum_digits補資料，"
                   "eg：1000筆 => 4位元，", type=int)
def load_dataset(
        old_dataset_name: str,
        new_dataset_name: str,
        output_dataset_name: str,
        mix_number: int,
        minimum_content_length: int,
        maximum_digits: int) -> None:
    """
    Create new dataset, that have some old dataset

    Args:
        maximum_digits:
        old_dataset_name(str): old datasets file name.
        new_dataset_name(str): new datasets file name.
        output_dataset_name(str): mix old and new datasets file name.
        mix_number(int): How many training sets to mix.
        minimum_content_length(int): Limit the minimum number of content when mix dataset.
    """

    Path(OLD_FOLDER).mkdir(exist_ok=True)
    Path(NEW_FOLDER).mkdir(exist_ok=True)

    old_dataset_path = Path(OLD_FOLDER, old_dataset_name).as_posix()
    new_dataset_path = Path(OLD_FOLDER, new_dataset_name).as_posix()
    output_dataset_path = Path(NEW_FOLDER, output_dataset_name).as_posix()

    logging.debug(f"old_dataset_path = {old_dataset_path}")
    logging.debug(f"new_dataset_path = {new_dataset_path}")
    logging.debug(f"output_dataset_path = {output_dataset_path}")
    logging.debug(f"mix_number = {mix_number}")
    logging.debug(f"minimum_content_length = {minimum_content_length}")

    old_dataset_df = pd.read_csv(old_dataset_path, sep="\t", names=NAMES)
    new_dataset_df = pd.read_csv(new_dataset_path, sep="\t", names=NAMES)

    if not mix_number:
        # 按照新資料集的比例進行取舊資料的比例
        # If the maximum number of digits exceeds three digits, the least common multiple will be used
        num_proportion = [new_dataset_df[new_dataset_df.label == 0].shape[0],
                          new_dataset_df[new_dataset_df.label == 1].shape[0],
                          new_dataset_df[new_dataset_df.label == 2].shape[0]]
        len_num = len(str(max(num_proportion)))

        if len_num > maximum_digits:
            max_digit = lcm(dataset_number=num_proportion)
        else:
            # max_digit = 10 ** (len(str(max(num_proportion))))
            max_digit = 10 ** maximum_digits
        num_proportion = 1 - (np.array(num_proportion) / max_digit)
        num_proportion = [int(round(num, 1) * max_digit) for num in num_proportion]
        logging.info(f"num_proportion = {num_proportion}")
        grab_mix_number(
            mix_number=num_proportion,
            old_training_df=old_dataset_df,
            new_training_df=new_dataset_df,
            output_dataset_path=output_dataset_path,
            minimum_content_length=minimum_content_length)
    else:
        mix_number_list = [mix_number] * 3  # 為了讓每個標籤都是固定的數量
        logging.info(f"mix_number_list = {mix_number_list}")
        grab_mix_number(
            mix_number=mix_number_list,
            old_training_df=old_dataset_df,
            new_training_df=new_dataset_df,
            output_dataset_path=output_dataset_path,
            minimum_content_length=minimum_content_length)


def combine_files(df_files: List[pd.DataFrame]):
    res = pd.concat(df_files, axis=0)
    res = res.sample(frac=1, axis=0)
    return res


def grab_mix_number(
        mix_number: List[int],
        old_training_df,
        new_training_df,
        output_dataset_path: str,
        minimum_content_length: int):
    """
    Grab the number of the dataset

    Args:
        mix_number(int): The mix number of the new training dataset.
        old_training_df(): old training dataset.
        new_training_df(): new training dataset.
        output_dataset_path(str): output datasets file path.
        minimum_content_length(int): Limit the minimum number of content when mix dataset.
    """
    mix_datasets = []
    for i, n in enumerate(mix_number):
        condition = (old_training_df.label == i) & (len(old_training_df.content) > minimum_content_length)
        mix_datasets.append(old_training_df[condition].sample(n=n, random_state=RANDOM_STATE))

    # 混合舊資料
    mix_datasets.append(new_training_df)
    mix_dataset = combine_files(mix_datasets)
    logging.info(f"0 total tag = {len(mix_dataset.query('label == 0'))}")
    logging.info(f"1 total tag = {len(mix_dataset.query('label == 1'))}")
    logging.info(f"2 total tag = {len(mix_dataset.query('label == 2'))}")
    mix_dataset.to_csv(output_dataset_path, sep="\t", index=False,
                       encoding="utf-8", header=False)


def lcm(dataset_number) -> int:
    """
    If the maximum number of digits exceeds three digits, the least common multiple will be used

    Args:
        dataset_number: The number of tags in the dataset
    """
    greater = max(dataset_number)
    while True:
        if greater % dataset_number[0] == 0 \
                and greater % dataset_number[1] == 0 \
                and greater % dataset_number[2] == 0:
            lcm = greater
            break
        greater += 1
    return lcm


if __name__ == '__main__':
    load_dataset()
