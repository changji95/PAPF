# Physics-constrained sequence learning with attention mechanism for multi-horizon production forecasting
This is the code for paper ["Physics-constrained sequence learning with attention mechanism for multi-horizon production forecasting"](https://www.sciencedirect.com/science/article/abs/pii/S2949891023009752) published in [Geoenergy Science and Engineering](https://www.sciencedirect.com/journal/geoenergy-science-and-engineering) (formerly known as [Journal of Petroleum Science and Engineering](https://www.sciencedirect.com/journal/journal-of-petroleum-science-and-engineering)). 

# Abstract
Production forecasting of oil and gas energy is a crucial task for reservoir management as it provides a fundamental basis for optimizing development plans and determining investment decisions. Recently, machine learning-based production forecasting methods are proliferating due to their powerful non-linear fitting capability. However, most existing methods lack direct consideration of the engineering physical processes, which may lead to production forecasts inconsistent with engineering science. In this paper, we propose a novel physics-constrained sequence learning method with attention mechanism for multi-horizon production forecasting. The proposed method is built upon on a backbone sequence-to-sequence neural network composed of gated recurrent unit (GRU) cells, which aims to learn the underlying decline behavior from historical production. An attention mechanism is introduced on the backbone network to improve the information utilization of historical production. Furthermore, known engineering parameters are incorporated into the decoder as physical constraints, enabling the network to take into consideration artificially controllable variables in addition to natural decline tendency. The proposed network is trained with a scheduled sampling strategy, which randomly allows the GRU decoder to access the true production from the previous step, thereby improving the robustness of the model to prediction errors. The proposed model is evaluated on the Volve public dataset. Experimental results validate the significant advantage of our method over ten baselines in multi-horizon production forecasting. In addition, ablation study from multiple aspects verifies the contributions of key components of our model. Our code is released at https://github.com/changji95/PAPF for academic use.

# Requirements
* python=3.7
* numpy=1.21
* pandas=1.3
* pytorch=1.4
* scikit-learn=1.0

GPU is recommended.

# Usage
Train: 
```
python main.py --fd 0 --layers 3 --nodes 160 --train
```
```
python main.py --fd 1 --layers 1 --nodes 80 --train
```
```
python main.py --fd 2 --layers 3 --nodes 160 --train
```
```
python main.py --fd 3 --layers 2 --nodes 80 --train
```
```
python main.py --fd 4 --layers 3 --nodes 120 --train
```
Test:
```
python main.py --fd 0 --layers 3 --nodes 160
```
```
python main.py --fd 1 --layers 1 --nodes 80
```
```
python main.py --fd 2 --layers 3 --nodes 160
```
```
python main.py --fd 3 --layers 2 --nodes 80
```
```
python main.py --fd 4 --layers 3 --nodes 120
```
Pretrained models for each fold are provided for testing.

# Citation
Please cite our paper if it's helpful to you in your research.
```
@article{chang2023physics,
  title={Physics-constrained sequence learning with attention mechanism for multi-horizon production forecasting},
  author={Chang, Ji and Zhang, Dongwei and Li, Yuling and Lv, Wenjun and Xiao, Yitian},
  journal={Geoenergy Science and Engineering},
  pages={212388},
  year={2023},
  publisher={Elsevier}
}
```
