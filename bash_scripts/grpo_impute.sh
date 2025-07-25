export CUDA_VISIBLE_DEVICES=1
export PORT=29501

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero2_singlegpu.yaml \
    --main_process_port $PORT \
    --num_processes=1 src/open_r1/grpo.py \
    --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_tldr.yaml