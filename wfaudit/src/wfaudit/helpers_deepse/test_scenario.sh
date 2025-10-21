WORKSPACE="/data/bcebere/http2/experiments/rw_baselines"
TESTCASE="seq_5_udemy"
N_WEBSITES=10
MODEL="df"
DEBUG="0"
N_TRACES=200

mkdir -p workspace
python preprocessing/create_dataset.py \
    --in_path ${WORKSPACE}/${TESTCASE}/tcp_repr/output_wefde \
    --out_path workspace/${TESTCASE}/dataset.npz \
    --n_websites ${N_WEBSITES} \
    --debug_mode ${DEBUG} \
    --n_traces $N_TRACES

python ./main.py \
    --data_path workspace/${TESTCASE}/dataset.npz \
    --n_traces $N_TRACES \
    --log_file workspace/${TESTCASE}/trace_${MODEL}.log \
    --results_file workspace/${TESTCASE}/results_${MODEL}.csv \
    --n_websites ${N_WEBSITES} \
    --k_fold 5 \
    --model ${MODEL} \
    --gpu_id 0 \
    --device cuda \
    --epochs 5 \
    --feature_length 500
