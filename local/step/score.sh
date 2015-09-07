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
# ++++++++++++++++++++++++++++++++++++++++++++++++++

. parse_options.sh || exit 1;

if [ $# != 3 ]; then
    echo "e.g.:  step02.sh 30k 100k nodup"
    exit 1;
fi

echo "exp:exp/${1}/${2}/${3}"
exp=exp/${1}/${2}/${3}
for x in ${exp}/*/decode*
do
    [ -d $x ] && grep Sum $x/score_*/*.sys | utils/best_wer.sh
done 2>/dev/null
