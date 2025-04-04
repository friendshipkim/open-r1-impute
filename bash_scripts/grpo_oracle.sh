ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero2_singlegpu.yaml \
    --main_process_port 29501 \
    --num_processes=1 src/open_r1/grpo.py \
    --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_chat_oracle.yaml