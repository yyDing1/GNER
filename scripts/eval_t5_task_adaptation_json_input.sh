set -x

port=$(shuf -i25000-30000 -n1)

BEAM_SIZE=1
MODEL_NAME_OR_PATH=output/flan-t5-xxl-task-adaptation/checkpoint-xxx
DATA_DIR=data
TEST_JSON_DIR=data/zero-shot-test.jsonl
DATA_CONFIG_DIR=configs/dataset_configs/task_adaptation_configs
INSTRUCTION_FILE=configs/instruction_configs/instruction.json
OUTPUT_DIR=output/flan-t5-xxl-task-adaptation-beam${BEAM_SIZE}

RUN_NAME=flan-t5-xxl-experiment

deepspeed --include="localhost:0,1,2,3,4,5,6,7" --master_port $port src/run.py \
    --bf16 True --tf32 True \
    --generation_num_beams ${BEAM_SIZE} \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_dir $DATA_DIR \
    --no_load_gner_customized_datasets \
    --test_json_dir $TEST_JSON_DIR \
    --preprocessing_num_workers 4 \
    --data_config_dir $DATA_CONFIG_DIR \
    --instruction_file $INSTRUCTION_FILE \
    --output_dir $OUTPUT_DIR \
    --per_device_eval_batch_size 4 \
    --run_name $RUN_NAME \
    --max_source_length 640 \
    --max_target_length 640 \
    --generation_max_length 640 \
    --overwrite_output_dir \
    --overwrite_cache \
    --seed 1234
