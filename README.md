# T2B-PE

The repo is the official implementation for the paper: Intriguing Properties of Positional Encoding in Time Series Forecasting.

## Usage 

1. Install Pytorch and other necessary dependencies.

```
pip install -r requirements.txt
```

1. The datasets can be obtained from the official “itransformer” repository or directly from [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/2ea5ca3d621e4e5ba36a/).

2. Run example: (Train, evaluate, and test on ECL dataset, lookback length:96, prediction length:96):
   ```
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --batch_size 16 \
  --learning_rate 0.0009048000000000001 \
  --weight_decay 9e-06\
  --use_weight_dec\
  --itr 1
   ```
