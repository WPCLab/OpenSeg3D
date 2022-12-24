#!/usr/bin/env bash

set -x

DATA_DIR=$1
WORK_DIR=$2
SPLIT=$3
PATHNAMES_FILE=$4


python3 -u $(dirname $0)/extract_image_feature.py $DATA_DIR $WORK_DIR $SPLIT $PATHNAMES_FILE