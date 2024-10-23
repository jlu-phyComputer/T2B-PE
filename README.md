# T2B-PE

![Static Badge](https://img.shields.io/badge/Time_Series_Forecasting-green)
![Static Badge](https://img.shields.io/badge/Positional_Embedding-Transformer-blue)
![Static Badge](https://img.shields.io/badge/to_be_continue-orange)

🥇 This repository contains the pytorch code for the arxiv paper: "Intriguing Properties of Positional Encoding in Time Series Forecasting".

## Model

![1713317818983](https://github.com/jlu-phyComputer/T2B-PE/assets/45681444/bd514017-1d60-4597-97ed-3b6d673ab899)


## Create Environment

Install Pytorch and other necessary dependencies.

```
pip install -r requirements.txt
```

## Data Availability

The datasets can be obtained from the official “itransformer” repository or directly from [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/2ea5ca3d621e4e5ba36a/).

## Run

Example: (Train, evaluate, and test on ECL dataset, lookback length:96, prediction length:196):

```
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_192 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --batch_size 16 \
  --learning_rate 0.00108 \
  --weight_decay 9e-06\
  --use_weight_dec\
  --pred_len 192
  --itr 1
```

The training command can be modified according to the above statement :)

## Cite

If you find our work and codes useful, please consider citing our paper and star our repository, thanks a lot.

```
@misc{zhang2024intriguing,
      title={Intriguing Properties of Positional Encoding in Time Series Forecasting}, 
      author={Jianqi Zhang and Jingyao Wang and Wenwen Qiang and Fanjiang Xu and Changwen Zheng and Fuchun Sun and Hui Xiong},
      year={2024},
      eprint={2404.10337},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

(arXiv version, the final version will be updated after the paper is published.)
