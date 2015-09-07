#!/bin/bash
# for construct the basic models below
# 1> mono phone GMM-HMM
# 2> tri1 phone GMM-HMM

. cmd.sh
. path.sh

set -e # exit on error, should be always here
# should run once only 
graph_dir=exp/tri1/graph_sw1_tg
false &&
{
    steps/train_mono.sh --nj 10 --cmd "$train_cmd" data/train_10k_nodup data/lang exp/mono 
    steps/align_si.sh --nj 10 --cmd "$train_cmd" data/train_30k_nodup data/lang exp/mono exp/mono_ali 
    steps/train_deltas.sh --cmd "$train_cmd" 3200 30000 data/train_30k_nodup data/lang exp/mono_ali exp/tri1 
    $train_cmd $graph_dir/mkgraph.log utils/mkgraph.sh data/lang_sw1_tg exp/tri1 $graph_dir
}

steps/decode_si.sh --nj 20 --cmd "$decode_cmd" --config conf/decode.config $graph_dir data/eval2000 exp/tri1/decode_eval2000_sw1_tg  
