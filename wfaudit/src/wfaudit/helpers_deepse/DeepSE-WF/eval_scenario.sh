testcase="cld3front_1_amazon"
N_WEBSITES=100
MODEL="df"
DEBUG="0"

python preprocessing/create_dataset.py \
    --in_path workspace/${testcase}/tcp_repr/output_wefde \
    --out_path workspace/${testcase}/dataset.npz \
    --n_websites ${N_WEBSITES} \
    --debug_mode ${DEBUG} \
    --n_traces 495

python ./main.py \
    --data_path workspace/${testcase}/dataset.npz \
    --n_traces 495 \
    --log_file workspace/${testcase}/trace_${MODEL}.log \
    --results_file workspace/${testcase}/results_${MODEL}.csv \
    --n_websites ${N_WEBSITES} \
    --k_fold 5 \
    --model ${MODEL}
    #--gpu_id 0 \
    #--device cuda \
    #--epochs 1 \
    #--feature_length 1000 \
