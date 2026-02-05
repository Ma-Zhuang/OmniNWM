config=$1
export find_unused_parameters=True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc-per-node $MLP_WORKER_GPU \
  --master-addr $MLP_WORKER_0_HOST \
  --master-port $MLP_WORKER_0_PORT \
  --node-rank $MLP_ROLE_INDEX \
  --nnodes $MLP_WORKER_NUM \
  tools/train.py \
  $config