#!/bin/bash
. cmd.sh
. path.sh
set -e # exit on error

mfccdir=mfcc
# for training data
steps/make_mfcc.sh --compress true --nj 20 --cmd "$train_cmd" data/train exp/make_mfcc/train $mfccdir
steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train $mfccdir 
utils/fix_data_dir.sh data/train 
# for evaluation data 
steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 data/eval2000 exp/make_mfcc/eval2000 $mfccdir
steps/compute_cmvn_stats.sh data/eval2000 exp/make_mfcc/eval2000 $mfccdir
utils/fix_data_dir.sh data/eval2000  # remove segments with problems

# for subset dof training data

utils/subset_data_dir.sh --first data/train 4000 data/train_dev # 5hr 6min
n=$[`cat data/train/segments | wc -l` - 4000]
echo "data/train_nodev is last ${n} data from data/train"
utils/subset_data_dir.sh --last data/train $n data/train_nodev


max_dup_num=200
utils/subset_data_dir.sh --shortest data/train_nodev 100000 data/train_100kshort

# remove duplicated utterance more 200 times
local/remove_dup_utts.sh ${max_dup_num} data/train_100kshort data/train_100kshort_nodup

utils/subset_data_dir.sh data/train_100kshort_nodup 10000 data/train_10k_nodup # be ware of the shortest 10k utt.

utils/subset_data_dir.sh --first data/train_nodev 30000 data/train_30k
local/remove_dup_utts.sh ${max_dup_num} data/train_30k data/train_30k_nodup  # 33hr

utils/subset_data_dir.sh --first data/train_nodev 100000 data/train_100k
local/remove_dup_utts.sh ${max_dup_num} data/train_100k data/train_100k_nodup  # 110hr

# remove duplicated utts of train_nodev, and construct the truely training dataset;
local/remove_dup_utts.sh 300 data/train_nodev data/train_nodup  # 286hr

