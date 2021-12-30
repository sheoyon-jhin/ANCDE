<h1 align='center'> Attentive Neural Controlled Differential Equations for Time-series Classification and Forecasting<br>
    for Time-Series Classification and Forecasting<br>
    [<a href="https://arxiv.org/abs/2109.01876">arXiv</a>] </h1>

### UEA_CharacterTrajectories
```
python3 uea_attentive.py --seed 2021 --missing_rate 0.3 --model="ancde" --h_channel 40 --hh_channel 40 --layer 3 --lr 0.001 --soft 'True' --slope_check '' --timewise 'True' --attention_channel 20 --attention_attention_channel 10 --step_mode 'valloss'
```

### PhysioNet Sepsis No OI
```
python3 sepsis_attentive.py --seed 2021 --intensity '' --model="ancde" --h_channel 49 --hh_channel 49 --layer 4 --lr 0.00001  --soft 'True' --slope_check '' --timewise 'True' --attention_channel 20 --attention_attention_channel 20 --step_mode 'valloss'
```

### PhysioNet Sepsis OI
```
python3 sepsis_attentive.py --seed 2021 --intensity 'True' --model="ancde" --h_channel 49 --hh_channel 49 --layer 4 --lr 0.00001  --soft 'True' --slope_check '' --timewise 'True' --attention_channel 20 --attention_attention_channel 20 --step_mode 'valloss'
```
### Google Stock
```
python3 stock.py --seed 2021 --sequence 25 --model="ancde_forecasting" --h_channel 12 --hh_channel 12 --layer 2 --lr 0.001 --soft '' --slope_check 'True' --timewise '' --attention_channel 4 --attention_attention_channel 8 --step_mode 'valloss'
```


### Citation
```bibtex
@article{jhin2021attentive,
  title={Attentive Neural Controlled Differential Equations for Time-series Classification and Forecasting},
  author={Jhin, Sheo Yon and Shin, Heejoo and Hong, Seoyoung and Park, Solhee and Park, Noseong},
  journal={arXiv preprint arXiv:2109.01876},
  year={2021}
}
```
