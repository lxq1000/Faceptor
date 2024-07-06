set -x

ROOT=../../../
export PYTHONPATH=$ROOT:$PYTHONPATH

port=50000
out_dir=<your_path>/output/faceptor/stage_1/


if [[ ! -d ${out_dir}"logs" ]]; then
  mkdir ${out_dir}"logs"
fi

now=$(date +"%Y%m%d_%H%M%S")

num_gpus=${1-4}
expname=${2-debug}
config=${expname}.yaml
start_time=${3-${now}}
load_iter=${4-newest}

log_file=${out_dir}logs/test_${expname}_${now}.log


CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=${num_gpus} --master_port=${port} ${ROOT}test.py \
    --config ${config} \
    --out_dir ${out_dir} \
    --expname ${expname} \
    --start_time ${start_time} \
    --load_iter ${load_iter} \
    --now ${now} \
    --port ${port} \
    2>&1 | tee ${log_file}