set -x

port=$(shuf -i25000-30000 -n1)

BEAM_SIZE=1
MODEL_NAME_OR_PATH=output/llama-7b-task-adaptation/checkpoint-xxx
DATA_DIR=data
DATA_CONFIG_DIR=configs/dataset_configs/task_adaptation_configs
INSTRUCTION_FILE=configs/instruction_configs/instruction.json
OUTPUT_DIR=output/llama-7b-task-adaptation-beam${BEAM_SIZE}

RUN_NAME=llama-7B-infer

deepspeed --include="localhost:0,1,2,3,4,5,6,7" --master_port $port src/run.py \
    --bf16 True --tf32 True \
    --generation_num_beams ${BEAM_SIZE} \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_dir $DATA_DIR \
    --preprocessing_num_workers 4 \
    --data_config_dir $DATA_CONFIG_DIR \
    --instruction_file $INSTRUCTION_FILE \
    --output_dir $OUTPUT_DIR \
    --per_device_eval_batch_size 4 \
    --run_name $RUN_NAME \
    --max_source_length 640 \
    --max_target_length 640 \
    --generation_max_length 1280 \
    --overwrite_output_dir \
    --overwrite_cache \
    --seed 1234
