#!/usr/bin/env bash

set -x

sudo bash /mnt/common/jianshu/liquidio/common-dataset-mount/common_dateset_mount.sh

tfrecord_file_list=$1
save_dir=$2
num_workers=$3
python3 -u $(dirname $0)/waymo_parser.py \
	--tfrecord_list_file=$tfrecord_file_list \
	--save_dir=$save_dir \
	--num_workers=$num_workers ${@:4}