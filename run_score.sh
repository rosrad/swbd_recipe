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

 ./local/step/score.sh  30k 100k all
 ./local/step/score.sh  30k 100k 100k
 ./local/step/score.sh  30k 30k 100k
 ./local/step/score.sh  30k 30k 30k
