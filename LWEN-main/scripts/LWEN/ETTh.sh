CUDA_VISIBLE_DEVICES=0
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=PDW

split_ratios=(
    "0.4 0.3 0.3"
    "0.1 0.3 0.6"
    #"0.4 0.4 0.2"
    #"0.2 0.5 0.3"
)

for win in 'hamming'  #'hann' 'blackman'
do
for win_size in 48 64
do
echo -e "\n=== Running with win_size=${win_size} ==="
for hid in 2 1.5
do
echo -e "\n=== Running with hid=${hid} ==="
for ratio in "${split_ratios[@]}"
do
echo -e "\n=== Running with split_ratio=${ratio} ==="


python -u ../../run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../../../DDDdata/iTransformer_datasets/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --enc_in 7 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --hidden_size 256 \
  --train_epochs 15 \
  --batch_size 16 \
  --patience 5 \
  --d_model 16 \
  --split_ratio ${ratio} \
  --hid_ratio $hid \
  --window_size $win_size \
  --window_type $win \
  --learning_rate 0.005 \
  --des 'Exp' \
  --itr 1 >logs/LongForecasting/"ETTh1_${win}_size${win_size}_hidd${hid}_$(echo ${ratio} | tr ' ' '_')_96_96.log"


python -u ../../run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../../../DDDdata/iTransformer_datasets/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --enc_in 7 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --hidden_size 256 \
  --train_epochs 15 \
  --batch_size 8 \
  --patience 5 \
  --d_model 16 \
  --split_ratio ${ratio} \
  --hid_ratio $hid \
  --window_size $win_size \
  --window_type $win \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1 >logs/LongForecasting/"ETTh1_${win}_size${win_size}_hidd${hid}_$(echo ${ratio} | tr ' ' '_')_96_192.log"

python -u ../../run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../../../DDDdata/iTransformer_datasets/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --enc_in 7 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --hidden_size 256 \
  --train_epochs 15 \
  --batch_size 64 \
  --patience 5 \
  --d_model 16 \
  --split_ratio ${ratio} \
  --hid_ratio $hid \
  --window_size $win_size \
  --window_type $win \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1 >logs/LongForecasting/"ETTh1_${win}_size${win_size}_hidd${hid}_$(echo ${ratio} | tr ' ' '_')_96_336.log"


python -u ../../run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../../../DDDdata/iTransformer_datasets/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --enc_in 7 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --hidden_size 256 \
  --train_epochs 15 \
  --batch_size 64 \
  --patience 5 \
  --d_model 16 \
  --split_ratio ${ratio} \
  --hid_ratio $hid \
  --window_size $win_size \
  --window_type $win \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1 >logs/LongForecasting/"ETTh1_${win}_size${win_size}_hidd${hid}_$(echo ${ratio} | tr ' ' '_')_96_720.log"


python -u ../../run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../../../DDDdata/iTransformer_datasets/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --enc_in 7 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --hidden_size 256 \
  --train_epochs 15 \
  --batch_size 16 \
  --patience 5 \
  --d_model 16 \
  --split_ratio ${ratio} \
  --hid_ratio $hid \
  --window_size $win_size \
  --window_type $win \
  --learning_rate 0.005 \
  --des 'Exp' \
  --itr 1 >logs/LongForecasting/"ETTh2_${win}_size${win_size}_hidd${hid}_$(echo ${ratio} | tr ' ' '_')_96_96.log"


python -u ../../run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../../../DDDdata/iTransformer_datasets/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --enc_in 7 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --hidden_size 128 \
  --train_epochs 15 \
  --batch_size 8 \
  --patience 5 \
  --d_model 16 \
  --split_ratio ${ratio} \
  --hid_ratio $hid \
  --window_size $win_size \
  --window_type $win \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1 >logs/LongForecasting/"ETTh2_${win}_size${win_size}_hidd${hid}_$(echo ${ratio} | tr ' ' '_')_96_192.log"

python -u ../../run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../../../DDDdata/iTransformer_datasets/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --enc_in 7 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --hidden_size 128 \
  --train_epochs 15 \
  --batch_size 64 \
  --patience 5 \
  --d_model 16 \
  --split_ratio ${ratio} \
  --hid_ratio $hid \
  --window_size $win_size \
  --window_type $win \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1 >logs/LongForecasting/"ETTh2_${win}_size${win_size}_hidd${hid}_$(echo ${ratio} | tr ' ' '_')_96_336.log"


python -u ../../run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../../../DDDdata/iTransformer_datasets/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --enc_in 7 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --hidden_size 256 \
  --train_epochs 15 \
  --batch_size 64 \
  --patience 5 \
  --d_model 16 \
  --split_ratio ${ratio} \
  --hid_ratio $hid \
  --window_size $win_size \
  --window_type $win \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1 >logs/LongForecasting/"ETTh2_${win}_size${win_size}_hidd${hid}_$(echo ${ratio} | tr ' ' '_')_96_720.log"

done
done
done
done