#!/bin/bash
set -e
DOCKER_IMAGE="gaussian-splatting:latest"

# Check for input arguments
if [[ $# -ne 2 ]]; then
    echo "Usage: run.bash <data_dir> <output_dir>" >&2
    exit 2
fi

DATA_DIR=$(realpath -s $1)
OUTPUT_DIR=$(realpath -s $2)

if [ ! -d ${DATA_DIR} ]; then
    echo "${DATA_DIR} is not a directory" >&2
    exit 2
fi

if [ ! -d ${OUTPUT_DIR} ]; then
    echo "Creating ${OUTPUT_DIR}" >&1
    mkdir -p ${OUTPUT_DIR}
fi


# Docker mounting + other flags
DOCKER_FLAGS="-it --rm --gpus all -v ${DATA_DIR}:${DATA_DIR} -v ${OUTPUT_DIR}:${OUTPUT_DIR} -e DATA_DIR -e OUTPUT_DIR"

# IO args for specifying data/model locations
GS_IO_ARGS="-s ${DATA_DIR} -m ${OUTPUT_DIR}"
GS_TRAIN_ARGS="${GS_IO_ARGS} --use_rgb --use_depth"
GS_TEST_ARGS="${GS_IO_ARGS} --skip_train"

if [ ! -f ${DATA_DIR}/points3d.ply ]; then
    echo "Initial pointclouds points3d.ply does not exist. Generating from depth map"
    path="${DATA_DIR}/scans/train/pcd"
    num_files=$(ls ${path} | wc -l)
    docker run ${DOCKER_FLAGS} ${DOCKER_IMAGE} python /gaussian-splatting/merge_point_cloud.py ${path} ${num_files} --output_file ${DATA_DIR}/points3d.ply
fi

# Train
docker run ${DOCKER_FLAGS} ${DOCKER_IMAGE} 
#python /gaussian-splatting/train.py ${RADEGS_TRAIN_ARGS}

# Test (render)
docker run ${DOCKER_FLAGS} ${DOCKER_IMAGE} python /gaussian-splatting/render.py ${RADEGS_TEST_ARGS}
