ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero2_singlegpu.yaml \
    --main_process_port 0 \
    --num_processes=1 src/open_r1/grpo.py \
    --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_chat.yaml