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

for pre in 96 192 336 720
do
python -u ../../run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../../../DDDdata/iTransformer_datasets/electricity/ \
  --data_path electricity.csv \
  --model_id electricity_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --enc_in 321 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pre \
  --embed_size 512 \
  --hidden_size 512 \
  --dropout 0 \
  --train_epochs 20 \
  --batch_size 4 \
  --patience 6 \
  --split_ratio ${ratio} \
  --hid_ratio $hid \
  --window_size $win_size \
  --window_type $win \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1 >logs/LongForecasting/"ECL_${win}_size${win_size}_hidd${hid}_$(echo ${ratio} | tr ' ' '_')_96_${pre}.log"
echo -e "\n=== done dataset:electricity   96-${pre} ==="

done
done
done
done
done