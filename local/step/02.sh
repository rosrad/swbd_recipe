#!/bin/bash

# for second steps to construct models below
# 1> tri2 (tri-phone GMM-HMM using alignments from tri1 model)
# 2> tri3 (tri-phone with LDA-MLLT applied GMM-HMM using alignments from tri1 model) 

set -e # exit on error
. cmd.sh
. path.sh # source the path.

echo "$0 $@"  # Print the command line for logging
# ++++++++++++++++++++++++++++++++++++++++++++++++++
# parameters initialized

stage=0
end=9 # max-steps should be less 9
# ++++++++++++++++++++++++++++++++++++++++++++++++++

. parse_options.sh || exit 1;

if [ $# != 3 ]; then
    echo "e.g.:  step02.sh 30k 100k nodup"
    exit 1;
fi

function level_guass() {
    if [ $# != 1 ]; then
        echo "no correct args"
        exit 1;
    fi
    case "$1" in 
        30k)
            echo "3200 30000"
            ;;
        60k)
            ;;
        100k)
            echo "5500 90000"
            ;;
        all)
            echo "11500 200000"
            ;;
    esac
}
# for parameterized tri-phones 
function adjust_name() {
    echo ${1/all/}_nodup |sed 's#^_*##g'
}

tr1=$(adjust_name ${1})
tr2=$(adjust_name ${2})
tr3=$(adjust_name ${3})
echo $tr1
echo $tr2
echo $tr3
# exit
# exp=exp/${1}/${2}/${3}          # 
echo "exp:exp/${1}/${2}/${3}"
# exit
# set +x
if [[ $stage -le 0  ]]; then
    if [ ! -e "exp/tri1_ali/ali.1.gz" ]; then
        steps/align_si.sh --nj 20 --cmd "$train_cmd" data/train_30k_nodup data/lang exp/tri1 exp/tri1_ali 
    fi 
    exp=exp/${1}
    if [ ! -e "${exp}/tri2/final.mdl" ]; then
        # echo "tri2 ${exp}"
        para=$(level_guass ${1})
        steps/train_deltas.sh --cmd "$train_cmd" ${para} data/train_${tr1} data/lang exp/tri1_ali ${exp}/tri2 
        echo "${para}" > ${exp}/tri2/para
        graph_dir=${exp}/tri2/graph_sw1_tg
        $train_cmd $graph_dir/mkgraph.log  utils/mkgraph.sh data/lang_sw1_tg ${exp}/tri2 $graph_dir
        steps/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode.config $graph_dir data/eval2000 ${exp}/tri2/decode_eval2000_sw1_tg
    fi
fi

if [[ $stage -le 1  ]]; then
    # on train_100k_nodup, which has 110hrs of data (start with the LDA+MLLT system)
    # Train tri3b, which is LDA+MLLT, on 100k_nodup data.
    pre_exp=exp/${1}

    if [ ! -e "${pre_exp}/tri2_ali_${tr2}/ali.1.gz" ]; then
        # echo "align tri2 ${pre_exp}"
        steps/align_si.sh --nj 20 --cmd "$train_cmd"  data/train_${tr2} data/lang ${pre_exp}/tri2 ${pre_exp}/tri2_ali_${tr2}
    fi
    exp=exp/${1}/${2}    
    para=$(level_guass ${2})
    if [ ! -e "${exp}/tri3b/final.mdl" ]; then
        steps/train_lda_mllt.sh --cmd "$train_cmd"  ${para} data/train_${tr2} data/lang ${pre_exp}/tri2_ali_${tr2} ${exp}/tri3b 
        echo "${para}" > ${exp}/tri3b/para
        graph_dir=${exp}/tri3b/graph_sw1_tg
        $train_cmd $graph_dir/mkgraph.log  utils/mkgraph.sh data/lang_sw1_tg ${exp}/tri3b $graph_dir
        steps/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode.config  $graph_dir data/eval2000 ${exp}/tri3b/decode_eval2000_sw1_tg
    fi
fi

if [[ $stage -le 2  ]]; then
    # Now train a LDA+MLLT+SAT model on the entire training data (train_nodup 286 hours)
    # Train tri4b, which is LDA+MLLT+SAT, on train_nodup data.
    pre_exp=exp/${1}/${2}
    if [ ! -e "${pre_exp}/tri3b_ali_${tr3}/ali.1.gz" ]; then
        steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" data/train_${tr3} data/lang ${pre_exp}/tri3b ${pre_exp}/tri3b_ali_${tr3} 
    fi
    exp=exp/${1}/${2}/${3}
    para=$(level_guass ${3})
    if [ ! -e  "${exp}/tri4b/final.mdl" ]; then
        steps/train_sat.sh  --cmd "$train_cmd" ${para}  data/train_${tr3} data/lang ${pre_exp}/tri3b_ali_${tr3} ${exp}/tri4b
        echo "${para}" > ${exp}/tri4b/para
        graph_dir=${exp}/tri4b/graph_sw1_tg
        $train_cmd $graph_dir/mkgraph.log  utils/mkgraph.sh data/lang_sw1_tg ${exp}/tri4b $graph_dir
        steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" --config conf/decode.config $graph_dir data/eval2000 ${exp}/tri4b/decode_eval2000_sw1_tg
        steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" --config conf/decode.config $graph_dir data/train_dev ${exp}/tri4b/decode_train_dev_sw1_tg
    fi
fi

# +++++++++++++++++++++++++++++++++++++++++++++
# for DNN steps
# +++++++++++++++++++++++++++++++++++++++++++++

exp=exp/${1}/${2}/${3}
gmmdir=${exp}/tri4b
data_fmllr=data-fmllr-tri4b/${exp}
# End of config.
if [[ $stage -le 3  ]]; then
    # Store fMLLR features, so we can train on them easily,
    # eval2000
    dir=$data_fmllr/eval2000
    if [ ! -e "$dir/feats.scp" ]; then
        steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
            --transform-dir $gmmdir/decode_eval2000_sw1_tg \
            $dir data/eval2000 $gmmdir $dir/log $dir/data || exit 1
    fi
    # train
    if [ ! -e "${exp}/tri4b_ali_${tr3}/ali.1.gz" ]; then
        steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" data/train_${tr3} data/lang ${gmmdir} ${gmmdir}_ali_${tr3} 
    fi
    dir=$data_fmllr/train_${tr3}
    if [ ! -e "$dir/feats.scp" ]; then
        steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
            --transform-dir ${gmmdir}_ali_${tr3} \
            $dir data/train_${tr3} $gmmdir $dir/log $dir/data || exit 1
        # split the data : 90% train 10% cross-validation (held-out)
        utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
    fi    
fi

# --- 2nd step ---
# for specially purpose of sync data between server and gpu server 

if [[ $stage -le 4  ]]; then
    # Pre-train DBN, i.e. a stack of RBMs
    dir=${exp}/dnn5b_pretrain-dbn
    # if [ ! -e "$dir/" ]; then
    (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log
    $cuda_cmd $dir/log/pretrain_dbn.log \
        steps/nnet/pretrain_dbn.sh --rbm-iter 1 $data_fmllr/train_${tr3} $dir || exit 1;
    # fi
fi

# --- 3rd step ---

if [[ $stage -le 5  ]]; then
    # Train the DNN optimizing per-frame cross-entropy.
    dir=${exp}/dnn5b_pretrain-dbn_dnn
    ali=${gmmdir}_ali_${tr3}
    feature_transform=${exp}/dnn5b_pretrain-dbn/final.feature_transform
    dbn=${exp}/dnn5b_pretrain-dbn/6.dbn
    (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
    # Train
    $cuda_cmd $dir/log/train_nnet.log \
        steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
        $data_fmllr/train_${tr3}_tr90 $data_fmllr/train_${tr3}_cv10 data/lang $ali $ali $dir || exit 1;
    # Decode (reuse HCLG graph)
    steps/nnet/decode.sh --nj 20 --cmd "$cuda_cmd" --config conf/decode_dnn.config --acwt 0.08333 \
        $gmmdir/graph_sw1_tg $data_fmllr/eval2000 $dir/decode_eval2000_sw1_tg || exit 1;
    # Rescore using unpruned trigram sw1
    steps/lmrescore.sh --mode 3 --cmd "$mkgraph_cmd" data/lang_sw1_tg data/lang_sw1_tg data/eval2000 \
        $dir/decode_eval2000_sw1_tg $dir/decode_eval2000_sw1_tg.3 || exit 1 
fi

# --- 4th step ---
# Sequence training using sMBR criterion, we do Stochastic-GD 
# with per-utterance updates. We use usually good acwt 0.1
# Lattices are re-generated after 1st epoch, to get faster convergence.
dir=${exp}/dnn5b_pretrain-dbn_dnn_smbr
srcdir=${exp}/dnn5b_pretrain-dbn_dnn
acwt=0.0909

if [[ $stage -le 6  ]]; then
    # First we generate lattices and alignments:
    steps/nnet/align.sh --nj 20 --cmd "$train_cmd" \
        $data_fmllr/train_${tr3} data/lang $srcdir ${srcdir}_ali || exit 1;
    steps/nnet/make_denlats.sh --nj 10 --sub-split 7 --cmd "$decode_cmd" --config conf/decode_dnn.config \
        --acwt $acwt $data_fmllr/train_${tr3} data/lang $srcdir ${srcdir}_denlats || exit 1;
fi

if [[ $stage -le 7  ]]; then
    
    # Re-train the DNN by 1 iteration of sMBR 
    steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 1 --acwt $acwt --do-smbr true \
        $data_fmllr/train_${tr3} data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
    # Decode (reuse HCLG graph)
    for ITER in 1; do
        steps/nnet/decode.sh --nj 20 --cmd "$cuda_cmd" --config conf/decode_dnn.config \
            --nnet $dir/${ITER}.nnet --acwt $acwt \
            $gmmdir/graph_sw1_tg $data_fmllr/eval2000 $dir/decode_eval2000_sw1_tg || exit 1;
        # Rescore using unpruned trigram sw1
        steps/lmrescore.sh --mode 3 --cmd "$mkgraph_cmd" data/lang_sw1_tg data/lang_sw1_tg data/eval2000 \
            $dir/decode_eval2000_sw1_tg $dir/decode_eval2000_sw1_tg.3 || exit 1 
    done 
fi
# for x in ${exp}/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
