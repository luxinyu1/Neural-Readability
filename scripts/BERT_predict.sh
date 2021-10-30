python ./neural_readability/predict.py \
        --arch 'bert' \
        --input-file-path './test.csv' \
        --model-ckpt-name 'checkpoint15' \
        --model-training-task 'newsela' \
        --bert-model-name 'bert-base-cased'