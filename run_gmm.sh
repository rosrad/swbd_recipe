#!/bin/bash

# This recipe is based on the run_edin.sh recipe, by Arnab Ghoshal,
# in the s5/ directory.
# This is supposed to be the "new" version of the switchboard recipe,
# after the s5/ one became a bit messy.  It is not 100% checked-through yet.

#exit 1;
# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.
# Caution: some of the graph creation steps use quite a bit of memory, so you
# should run this on a machine that has sufficient memory.

. cmd.sh
. path.sh
set -e # exit on error
steps/align_si.sh --nj 10 --cmd "$train_cmd" data/train_30k_nodup data/lang exp/mono exp/mono_ali 
steps/train_deltas.sh --cmd "$train_cmd" 3200 30000 data/train_30k_nodup data/lang exp/mono_ali exp/tri1 

graph_dir=exp/tri1/graph_sw1_tg
$train_cmd $graph_dir/mkgraph.log utils/mkgraph.sh data/lang_sw1_tg exp/tri1 $graph_dir
steps/decode_si.sh --nj 20 --cmd "$decode_cmd" --config conf/decode.config $graph_dir data/eval2000 exp/tri1/decode_eval2000_sw1_tg  

steps/align_si.sh --nj 20 --cmd "$train_cmd" data/train_30k_nodup data/lang exp/tri1 exp/tri1_ali 
steps/train_deltas.sh --cmd "$train_cmd" 3200 30000 data/train_30k_nodup data/lang exp/tri1_ali exp/tri2 

graph_dir=exp/tri2/graph_sw1_tg
$train_cmd $graph_dir/mkgraph.log  utils/mkgraph.sh data/lang_sw1_tg exp/tri2 $graph_dir
steps/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode.config $graph_dir data/eval2000 exp/tri2/decode_eval2000_sw1_tg


# on train_100k_nodup, which has 110hrs of data (start with the LDA+MLLT system)
# Train tri3b, which is LDA+MLLT, on 100k_nodup data.

steps/align_si.sh --nj 20 --cmd "$train_cmd"  data/train_100k_nodup data/lang exp/tri2 exp/tri2_ali_100k_nodup 
steps/train_lda_mllt.sh --cmd "$train_cmd"  5500 90000 data/train_100k_nodup data/lang exp/tri2_ali_100k_nodup exp/tri3b 

graph_dir=exp/tri3b/graph_sw1_tg
$train_cmd $graph_dir/mkgraph.log  utils/mkgraph.sh data/lang_sw1_tg exp/tri3b $graph_dir
steps/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode.config  $graph_dir data/eval2000 exp/tri3b/decode_eval2000_sw1_tg


# ------------- 100 k --------------------------
steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" data/train_100k_nodup data/lang exp/tri3b exp/tri3b_ali_100k_nodup 
steps/train_sat.sh  --cmd "$train_cmd"  5500 90000 data/train_100k_nodup data/lang exp/tri3b_ali_100k_nodup  exp/tri4a 

graph_dir=exp/tri4a/graph_sw1_tg
$train_cmd $graph_dir/mkgraph.log  utils/mkgraph.sh data/lang_sw1_tg exp/tri4a $graph_dir
steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" --config conf/decode.config  $graph_dir data/eval2000 exp/tri4a/decode_eval2000_sw1_tg

# MMI training starting from the LDA+MLLT+SAT systems on the train_100k_nodup (110hr) set
steps/align_fmllr.sh --nj 10 --cmd "$train_cmd"  data/train_100k_nodup data/lang exp/tri4a exp/tri4a_ali_100k_nodup || exit 1
steps/make_denlats.sh --nj 10 --cmd "$decode_cmd" --config conf/decode.config --transform-dir exp/tri4a_ali_100k_nodup \
#  data/train_100k_nodup data/lang exp/tri4a exp/tri4a_denlats_100k_nodup 

num_mmi_iters=4
steps/train_mmi.sh --cmd "$decode_cmd" --boost 0.1 --num-iters $num_mmi_iters  data/train_100k_nodup data/lang exp/tri4a_{ali,denlats}_100k_nodup  exp/tri4a_mmi_b0.1 

for iter in 1 2 3 4; do
    graph_dir=exp/tri4a/graph_sw1_tg
    decode_dir=exp/tri4a_mmi_b0.1/decode_eval2000_${iter}.mdl_sw1_tg
    steps/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode.config \
    --iter $iter --transform-dir exp/tri4a/decode_eval2000_sw1_tg  $graph_dir data/eval2000 $decode_dir
done



# Now do fMMI+MMI training
steps/train_diag_ubm.sh --silence-weight 0.5 --nj 10 --cmd "$train_cmd" \
  700 data/train_100k_nodup data/lang exp/tri4a_ali_100k_nodup exp/tri4a_dubm
    
steps/train_mmi_fmmi.sh --learning-rate 0.005 --boost 0.1 --cmd "$train_cmd" \
  data/train_100k_nodup data/lang exp/tri4a_ali_100k_nodup exp/tri4a_dubm \
  exp/tri4a_denlats_100k_nodup exp/tri4a_fmmi_b0.1 

 
for iter in 4 5 6 7 8; do
      graph_dir=exp/tri4a/graph_sw1_tg
      decode_dir=exp/tri4a_fmmi_b0.1/decode_eval2000_it${iter}_sw1_tg
      steps/decode_fmmi.sh --nj 20 --cmd "$decode_cmd" --iter $iter \
	--transform-dir exp/tri4a/decode_eval2000_sw1_tg \
	--config conf/decode.config $graph_dir data/eval2000 $decode_dir
done  


# ------------- 192 k -------------------------- 
#steps/lmrescore.sh --mode 3 --cmd "$mkgraph_cmd" data/lang_sw1_fsh_tgpr data/lang_sw1_fsh_tg data/eval2000 \
#  exp/tri4b/decode_eval2000_sw1_fsh_tgpr exp/tri4b/decode_eval2000_sw1_fsh_tg.3 || exit 1

# Now train a LDA+MLLT+SAT model on the entire training data (train_nodup 286 hours)
# Train tri4b, which is LDA+MLLT+SAT, on train_nodup data.
steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" data/train_nodup data/lang exp/tri3b exp/tri3b_ali_nodup 
steps/train_sat.sh  --cmd "$train_cmd"  11500 200000 data/train_nodup data/lang exp/tri3b_ali_nodup exp/tri4b

graph_dir=exp/tri4b/graph_sw1_tg
$train_cmd $graph_dir/mkgraph.log  utils/mkgraph.sh data/lang_sw1_tg exp/tri4b $graph_dir
steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" --config conf/decode.config $graph_dir data/eval2000 exp/tri4b/decode_eval2000_sw1_tg
steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" --config conf/decode.config $graph_dir data/train_dev exp/tri4b/decode_train_dev_sw1_tg
 
# MMI training starting from the LDA+MLLT+SAT systems on the train_nodup (286hr) set 
steps/align_fmllr.sh --nj 15 --cmd "$train_cmd"  data/train_nodup data/lang exp/tri4b exp/tri4b_ali_nodup || exit 1
steps/make_denlats.sh --nj 15 --cmd "$decode_cmd" --config conf/decode.config --transform-dir exp/tri4b_ali_nodup \
  data/train_nodup data/lang exp/tri4b exp/tri4b_denlats_nodup 
num_mmi_iters=4
steps/train_mmi.sh --cmd "$decode_cmd" --boost 0.1 --num-iters $num_mmi_iters  data/train_nodup data/lang exp/tri4b_{ali,denlats}_nodup exp/tri4b_mmi_b0.1 
for iter in 1 2 3 4; do
      graph_dir=exp/tri4b/graph_sw1_tg
      decode_dir=exp/tri4b_mmi_b0.1/decode_eval2000_${iter}.mdl_sw1_tg
      steps/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode.config \
    --iter $iter --transform-dir exp/tri4b/decode_eval2000_sw1_tg $graph_dir data/eval2000 $decode_dir   
done

# Now do fMMI+MMI training
steps/train_diag_ubm.sh --silence-weight 0.5 --nj 15 --cmd "$train_cmd" 700 data/train_nodup data/lang exp/tri4b_ali_nodup exp/tri4b_dubm
 
steps/train_mmi_fmmi.sh --learning-rate 0.005 --boost 0.1 --cmd "$train_cmd" \
  data/train_nodup data/lang exp/tri4b_ali_nodup exp/tri4b_dubm exp/tri4b_denlats_nodup exp/tri4b_fmmi_b0.1  

for iter in 4 5 6 7 8; do
      graph_dir=exp/tri4b/graph_sw1_tg
      decode_dir=exp/tri4b_fmmi_b0.1/decode_eval2000_it${iter}_sw1_tg
      steps/decode_fmmi.sh --nj 20 --cmd "$decode_cmd" --iter $iter \
		--transform-dir exp/tri4b/decode_eval2000_sw1_tg \
		--config conf/decode.config $graph_dir data/eval2000 $decode_dir
done

steps/train_sat.sh --cmd "$train_cmd" \
  4000 100000 data/train_100k_nodup data/lang exp/tri4a_ali_100k_nodup exp/tri5a || exit 1;
(
  utils/mkgraph.sh data/lang_test exp/tri5a exp/tri5a/graph || exit 1;
  steps/decode_fmllr.sh --cmd "$decode_cmd" --config conf/decode.config \
   --nj 30 exp/tri5a/graph data/eval2000 exp/tri5a/decode_eval2000 || exit 1;
  steps/decode_fmllr.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri5a/graph data/train_dev exp/tri5a/decode_train_dev || exit 1;
)
