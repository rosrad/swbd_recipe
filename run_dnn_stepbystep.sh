#!/bin/bash

# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains a DNN on top of fMLLR features. 
# The training is done in 3 stages,
#
# 1) RBM pre-training:
#    in this unsupervised stage we train stack of RBMs, 
#    a good starting point for frame cross-entropy trainig.
# 2) frame cross-entropy training:
#    the objective is to classify frames to correct pdfs.
# 3) sequence-training optimizing sMBR: 
#    the objective is to emphasize state-sequences with better 
#    frame accuracy w.r.t. reference alignment.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

# Config:
gmmdir=exp/tri4b
data_fmllr=data-fmllr-tri4b
stage=0 # resume training with --stage=N
# End of config.
. utils/parse_options.sh || exit 1;
#

# --- 1st step ---

#beta-order-> if [ $stage -le 0 ]; then
#beta-order->   # Store fMLLR features, so we can train on them easily,
#beta-order->   # eval2000
#beta-order->   dir=$data_fmllr/eval2000
#beta-order->   steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
#beta-order->      --transform-dir $gmmdir/decode_eval2000_sw1_tg \
#beta-order->      $dir data/eval2000 $gmmdir $dir/log $dir/data || exit 1
#beta-order->   # train
#beta-order->   dir=$data_fmllr/train_nodup
#beta-order->   steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
#beta-order->      --transform-dir ${gmmdir}_ali_nodup \
#beta-order->      $dir data/train_nodup $gmmdir $dir/log $dir/data || exit 1
#beta-order->   # split the data : 90% train 10% cross-validation (held-out)
#beta-order->   utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
#beta-order-> fi

# --- 2nd step ---

#beta-order->if [ $stage -le 1 ]; then
#beta-order->  # Pre-train DBN, i.e. a stack of RBMs
#beta-order->  dir=exp/dnn5b_pretrain-dbn
#beta-order->  (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log
#beta-order->  $cuda_cmd $dir/log/pretrain_dbn.log \
#beta-order->    steps/nnet/pretrain_dbn.sh --rbm-iter 1 $data_fmllr/train_nodup $dir || exit 1;
#beta-order->fi

# --- 3rd step ---

#beta-order->if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
#beta-order->  dir=exp/dnn5b_pretrain-dbn_dnn
#beta-order->  ali=${gmmdir}_ali_nodup
#beta-order->  feature_transform=exp/dnn5b_pretrain-dbn/final.feature_transform
#beta-order->  dbn=exp/dnn5b_pretrain-dbn/6.dbn
#beta-order->  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
#beta-order->  # Train
#beta-order->  $cuda_cmd $dir/log/train_nnet.log \
#beta-order->    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
#beta-order->    $data_fmllr/train_nodup_tr90 $data_fmllr/train_nodup_cv10 data/lang $ali $ali $dir || exit 1;
#beta-order->  # Decode (reuse HCLG graph)
#beta-order->  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.08333 \
#beta-order->    $gmmdir/graph_sw1_tg $data_fmllr/eval2000 $dir/decode_eval2000_sw1_tg || exit 1;
#beta-order->  # Rescore using unpruned trigram sw1
#beta-order->  steps/lmrescore.sh --mode 3 --cmd "$mkgraph_cmd" data/lang_sw1_tg data/lang_sw1_tg data/eval2000 \
#beta-order->    $dir/decode_eval2000_sw1_tg $dir/decode_eval2000_sw1_tg.3 || exit 1 
#beta-order->fi

# --- 4th step ---

# Sequence training using sMBR criterion, we do Stochastic-GD 
# with per-utterance updates. We use usually good acwt 0.1
# Lattices are re-generated after 1st epoch, to get faster convergence.
#beta-order->dir=exp/dnn5b_pretrain-dbn_dnn_smbr
#beta-order->srcdir=exp/dnn5b_pretrain-dbn_dnn
#beta-order->acwt=0.0909

#beta-order->if [ $stage -le 3 ]; then
#beta-order->  # First we generate lattices and alignments:
#beta-order->  steps/nnet/align.sh --nj 20 --cmd "$train_cmd" \
#beta-order->    $data_fmllr/train_nodup data/lang $srcdir ${srcdir}_ali || exit 1;
#beta-order->  steps/nnet/make_denlats.sh --nj 10 --sub-split 7 --cmd "$decode_cmd" --config conf/decode_dnn.config \
#beta-order->    --acwt $acwt $data_fmllr/train_nodup data/lang $srcdir ${srcdir}_denlats || exit 1;
#beta-order->fi

#beta-order->if [ $stage -le 4 ]; then
#beta-order->  # Re-train the DNN by 1 iteration of sMBR 
#beta-order->  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 1 --acwt $acwt --do-smbr true \
#beta-order->    $data_fmllr/train_nodup data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
#beta-order->  # Decode (reuse HCLG graph)
#beta-order->  for ITER in 1; do
#beta-order->    steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config \
#beta-order->      --nnet $dir/${ITER}.nnet --acwt $acwt \
#beta-order->      $gmmdir/graph_sw1_tg $data_fmllr/eval2000 $dir/decode_eval2000_sw1_tg || exit 1;
#beta-order->    # Rescore using unpruned trigram sw1
#beta-order->    steps/lmrescore.sh --mode 3 --cmd "$mkgraph_cmd" data/lang_sw1_tg data/lang_sw1_tg data/eval2000 \
#beta-order->      $dir/decode_eval2000_sw1_tg $dir/decode_eval2000_sw1_tg.3 || exit 1 
#beta-order->  done 
#beta-order->fi

# --- 5th step ---

# Re-generate lattices, run 4 more sMBR iterations
dir=exp/dnn5b_pretrain-dbn_dnn_smbr_i1lats
srcdir=exp/dnn5b_pretrain-dbn_dnn_smbr
acwt=0.0909

#beta-order->if [ $stage -le 5 ]; then
#beta-order->  # First we generate lattices and alignments:
#beta-order->  steps/nnet/align.sh --nj 20 --cmd "$train_cmd" \
#beta-order->    $data_fmllr/train_nodup data/lang $srcdir ${srcdir}_ali || exit 1;
#beta-order->  steps/nnet/make_denlats.sh --nj 10 --sub-split 7 --cmd "$decode_cmd" --config conf/decode_dnn.config \
#beta-order->    --acwt $acwt $data_fmllr/train_nodup data/lang $srcdir ${srcdir}_denlats || exit 1;
#beta-order->fi

# --- 6th step ---

if [ $stage -le 6 ]; then
  # Re-train the DNN by 1 iteration of sMBR 
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 2 --acwt $acwt --do-smbr true \
    $data_fmllr/train_nodup data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  # Decode (reuse HCLG graph)
  for ITER in 1 2; do
    steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmmdir/graph_sw1_tg $data_fmllr/eval2000 $dir/decode_eval2000_sw1_tg || exit 1;
    # Rescore using unpruned trigram sw1
    steps/lmrescore.sh --mode 3 --cmd "$mkgraph_cmd" data/lang_sw1_tg data/lang_sw1_tg data/eval2000 \
      $dir/decode_eval2000_sw1_tg $dir/decode_eval2000_sw1_tg.3 || exit 1 
  done 
fi

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
