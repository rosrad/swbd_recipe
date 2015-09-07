#!/bin/bash
. cmd.sh
. path.sh
set -e # exit on error
# swbd=/home/renbo/Corpus/swbd/
# tr_data=${swbd}/Data/Training_Data/
# ev_data=${swbd}/Data/Eval_Data/

local/swbd1_data_prep.sh ${tr_data}
local/swbd1_prepare_dict.sh
utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang

fisher_opt=
local/swbd1_train_lms.sh $fisher_opt data/local/train/text data/local/dict/lexicon.txt data/local/lm

srilm_opts="-subset -prune-lowprobs -unk -tolower -order 3"
LM=data/local/lm/sw1.o3g.kn.gz
for l in lang_sw1_tg lang_sw1_fsh_tg lang_sw1_fsh_tgpr; do
    utils/format_lm_sri.sh --srilm-opts "$srilm_opts" data/lang $LM data/local/dict/lexicon.txt data/${l}
done

local/eval2000_data_prep.sh ${ev_data}/eval2000/2000HUB5 ${ev_data}/eval2000/2000_hub5_eng_eval_tr

