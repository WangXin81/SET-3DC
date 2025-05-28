# SET-3DC
This repository contains the codebase for SET-3DC, implemented using PyTorch for conducting experiments on SECOND and HRSCD.
## Data preparation:
```
train  
  --A  
  --B 
  --label1  
  --label2  
val  
  --A 
  --B 
  --label1  
  --label2  
test  
  --A
  --B 
  --label1  
  --label2  
```
## FIGS        

![image](/figs/fig1.jpg)

## Datasets

SECOND:
[https://drive.google.com/file/d/1QlAdzrHpfBIOZ6SK78yHF2i1u6tikmBc/view?usp=sharing]

HRSCD:
[https://rcdaudt.github.io/hrscd/]

## Environments

1.CUDA  11.8

2.Pytorch 2.4

3.Python 3.10

## Train and Test
The training and evaling pipeline is organized in train.py and Eval_SCD.py.
```bash
python train.py
```

```bash
python Eval_SCD.py
```

