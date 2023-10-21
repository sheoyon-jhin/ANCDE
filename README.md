<h1 align='center'> Attentive Neural Controlled Differential Equations for Time-series Classification and Forecasting<br>
    for Time-Series Classification and Forecasting<br>
    [<a href="https://arxiv.org/abs/2109.01876">ICDM 2021</a>] </h1>
Neural networks inspired by differential equations have proliferated for the past several years. Neural ordinary differential equations (NODEs) and neural controlled differential equations (NCDEs) are two representative examples of them. In theory, NCDEs provide better representation learning capability for time-series data than NODEs. In particular, it is known that NCDEs are suitable for processing irregular time-series data. Whereas NODEs have been successfully extended after adopting attention, however, it had not been studied yet how to integrate attention into NCDEs. To this end, we present the method of Attentive Neural Controlled Differential Equations (ANCDEs) for time-series classification and forecasting, where dual NCDEs are used: one for generating attention values, and the other for evolving hidden vectors for a downstream machine learning task. We conduct experiments with three real-world time-series datasets and 10 baselines. After dropping some values, we also conduct irregular time-series experiments. Our method consistently shows the best accuracy in all cases by non-trivial margins. Our visualizations also show that the presented attention mechanism works as intended by focusing on crucial information.
<p align="center">
  
  <img align="middle" src="./imgs/Neuralcde.png" alt="NeuralCDE"/> 
  The overall architecture of NCDE
</p>

<p align="center">
  <img align="middle" src="./imgs/AttentiveNeuralCDE.png" alt="ANCDE" /> 
  Our proposed attentive NCDE In our method, the bottom NCDE produces attention values and the top NCDE produces the last hidden vector. 
  
</p>
<p align="center">
  <img align="middle" src="./imgs/ANCDE_flow.png" alt="ANCDE_model_structure in detail"/> 
  The detailed architecture of our proposed attentive NCDE. From the raw time-series data x(t), the continuous path X(t) is created by the natural cubic spline algorithm. The bottom NCDE reads X(t) to produce attention values. The path Y(t) is defined as the (element-wise) multiplication of X(t) and the attention value at time t. The top NCDE produces the last hidden vector with Y(t). Our framework supports both classification and regression.
</p>
<p align="center">
  <img align="center" src="./imgs/sepsis.png" alt="PhysioNet Sepsis attention visualization" width="400"/> 
  
</p>
<p>The soft attention visualization in physioNet Sepsis. Blue means strong attention and white means weak attention. Note that the bottom NCDE gives strong attention when there are non-trivial changes on the curves.</p>

<p align="center">
  <img align="center" src="./imgs/stock.png" alt="Google Stock attention visualization" width="400"/> 
  
</p>

  <p>The soft attention visualization in Google Stock. Blue means strong attention and white means weak attention. Note that the open price has weak attention always. This is because it shows high correlations to other values. Our attention model strategically ignores the open price.</p>

### create conda environments
```
conda env create --file ancde.yml
```

### activate conda 
```
conda activate neuralcde
```

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

### Google Stock(single-step forecasting)
```
python3 stock.py --seed 2021 --input_seq 24 --output_seq 1 --model="ancde_forecasting" --h_channel 12 --hh_channel 12 --layer 2 --lr 0.001 --soft '' --slope_check 'True' --timewise '' --attention_channel 4 --attention_attention_channel 8 --step_mode 'valloss'
```

### Google Stock(multi-step forecasting)
```
python3 stock.py --seed 2021 --input_seq 20 --output_seq 5 --model="ancde_forecasting" --h_channel 15 --hh_channel 15 --layer 2 --lr 0.005 --soft '' --slope_check 'True' --timewise '' --attention_channel 4 --attention_attention_channel 8 --step_mode 'valloss'
```

### Mujoco 
```
python3 mujoco.py --seed 2021 --input_seq 20 --output_seq 5  --model="ancde_forecasting" --h_channel 12 --hh_channel 12 --layer 2 --lr 0.001 --soft '' --slope_check 'True' --timewise '' --attention_channel 4 --attention_attention_channel 8 --step_mode 'valloss'
```

### Run shell file (ex PhysioNet Sepsis dataset)
```
sh sepsis.sh 
```
### Run shell file, background  (ex PhysioNet Sepsis dataset)
```
nohup sh sepsis.sh 2>sepsis.err 1>sepsis.log & 
```

### Citation
```bibtex
@article{jhin2021attentive,
  title={Attentive Neural Controlled Differential Equations for Time-series Classification and Forecasting},
  author={Jhin, Sheo Yon and Shin, Heejoo and Hong, Seoyoung and Jo, Minju and Park, Solhee and Park, Noseong},
  journal={ICDM},
  year={2021}
}
```
