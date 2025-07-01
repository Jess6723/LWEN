export CUDA_VISIBLE_DEVICES=0

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

for ratio in "${split_ratios[@]}"
do

for hid in 1.5 2
do
echo -e "\n=== Running with win_size=${win_size}  split_ratio=${ratio} hid=${hid} ==="

python -u ../../run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../../../DDDdata/iTransformer_datasets/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --enc_in 862 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --embed_size 256 \
  --hidden_size 512 \
  --dropout 0 \
  --train_epochs 20 \
  --batch_size 16 \
  --patience 6 \
  --split_ratio ${ratio} \
  --hid_ratio $hid \
  --window_size $win_size \
  --window_type $win \
  --learning_rate 0.005 \
  --des 'Exp' \
  --itr 1 >logs/LongForecasting/"traffic_${win}_size${win_size}_hidd${hid}_$(echo ${ratio} | tr ' ' '_')_96_96.log"
echo -e "\n=== done dataset:traffic   96-96 ==="

python -u ../../run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../../../DDDdata/iTransformer_datasets/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --enc_in 862 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --embed_size 256 \
  --hidden_size 512 \
  --dropout 0 \
  --train_epochs 20 \
  --batch_size 16 \
  --patience 6 \
  --split_ratio ${ratio} \
  --hid_ratio $hid \
  --window_size $win_size \
  --window_type $win \
  --learning_rate 0.005 \
  --des 'Exp' \
  --itr 1 >logs/LongForecasting/"traffic_${win}_size${win_size}_hidd${hid}_$(echo ${ratio} | tr ' ' '_')_96_192.log"
echo -e "\n=== done dataset:traffic   96-192 ==="

python -u ../../run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../../../DDDdata/iTransformer_datasets/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --enc_in 862 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --embed_size 256 \
  --hidden_size 512 \
  --dropout 0 \
  --train_epochs 20 \
  --batch_size 16 \
  --patience 6 \
  --split_ratio ${ratio} \
  --hid_ratio $hid \
  --window_size $win_size \
  --window_type $win \
  --learning_rate 0.005 \
  --des 'Exp' \
  --itr 1 >logs/LongForecasting/"traffic_${win}_size${win_size}_hidd${hid}_$(echo ${ratio} | tr ' ' '_')_96_336.log"
echo -e "\n=== done dataset:traffic   96-336 ==="

python -u ../../run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../../../DDDdata/iTransformer_datasets/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --enc_in 862 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --embed_size 256 \
  --hidden_size 1024 \
  --dropout 0 \
  --train_epochs 20 \
  --batch_size 16 \
  --patience 6 \
  --split_ratio ${ratio} \
  --hid_ratio $hid \
  --window_size $win_size \
  --window_type $win \
  --learning_rate 0.005 \
  --des 'Exp' \
  --itr 1 >logs/LongForecasting/"traffic_${win}_size${win_size}_hidd${hid}_$(echo ${ratio} | tr ' ' '_')_96_720.log"
echo -e "\n=== done dataset:traffic   96-720 ==="

done
done
done
done