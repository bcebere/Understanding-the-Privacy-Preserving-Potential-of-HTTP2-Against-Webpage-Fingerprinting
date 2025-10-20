WORKSPACE="" #REDACTED
N_WEBSITES=100
MODEL="df"
FEAT_LEN=5000
setup="tcp_repr"
DEBUG=0
FOLDS=3
N_TRACES=500

for experiment in `ls $WORKSPACE`
do
    if [ ! -d $WORKSPACE/$experiment ]
    then
        continue
    fi
    if [ "$experiment" = "mocks" ]
    then
        continue
    fi

    testcase=$experiment
    testcase_workspace=$WORKSPACE/$experiment/$setup/eval_deepse
    mkdir -p $testcase_workspace

    leak_input="`ls -A $WORKSPACE/$experiment/$setup/output_ml/X_1C_multi.npy 2>/dev/null | wc -l`"
    if [ $leak_input -eq 0 ]
    then
        #echo "$experiment not generated yet"
        continue
    fi

    for testtype in "real" "sanity_0"
    do
        internal_scenario=${testcase_workspace}/${testtype}
        if [ -f ${internal_scenario}/results_${MODEL}.csv ]
        then
            mv ${internal_scenario}/results_${MODEL}.csv  ${internal_scenario}/zz.results_${MODEL}.csv
        fi

        leak_eval="`ls -A ${internal_scenario}/results_hl_${MODEL}.csv 2>/dev/null | wc -l`"
        if [ $leak_eval -ne 0 ]
        then
            echo "$WORKSPACE/$experiment/$setup DEBUG=$DEBUG done"
            continue
        fi



        internal_scenario=${testcase_workspace}/${testtype}
        echo "Eval $testcase_workspace/${testtype} DEBUG=$DEBUG"

        if [ ! -f ${internal_scenario}/dataset.npz ]
        then
            python preprocessing/create_dataset.py \
                --in_path ${WORKSPACE}/${testcase}/tcp_repr/output_wefde \
                --out_path ${internal_scenario}/dataset.npz \
                --n_websites ${N_WEBSITES} \
                --debug_mode ${DEBUG} \
                --feature_length ${FEAT_LEN} \
                --n_traces ${N_TRACES}
        fi

        python ./main.py \
            --data_path ${internal_scenario}/dataset.npz \
            --n_traces ${N_TRACES} \
            --log_file ${internal_scenario}/trace_${MODEL}.log \
            --results_file ${internal_scenario}/results_hl_${MODEL}.csv \
            --n_websites ${N_WEBSITES} \
            --k_fold ${FOLDS} \
            --feature_length ${FEAT_LEN} \
            --gpu_id 0 \
            --device cuda \
            --model ${MODEL}
    done
done
