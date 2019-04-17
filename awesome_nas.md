## Architecture Search

### Labels
- **Optimization Method**
    - `RL`--> **R**einforcement **L**earning
    - `EA`--> **E**volution **A**lgorithm
    - `GD`--> **G**radient **D**escent
    - `BO`--> **B**ayesian **O**ptimisation
    - `MCTS` --> **M**onte **C**arlo **T**ree **S**earch
    - `SMBO` --> **S**equential **M**odel-**B**ased **O**ptimization
    - `1S` --> **1**-**S**hot Learning
- **Objective Function**
    - `DE` --> **DE**vice-related: inference time, memory usage, power consumption
- **Training**
    - `FT` --> **F**ine**T**une on pretrained models
    - `SR` --> **S**c**R**atch
    - `TL` --> **T**ransfer **L**earning between tasks
    - `PS` --> **P**arameter **S**haring
    - `NM` --> **N**etwork **M**orphisms
    - `KT` --> Knowledge Transfer
- **Search Level**
    - `TP` --> **T**o**P**ology of connection paths
    - `SG` --> **S**ub**G**raph within a large computational graph
    - `SM` --> Frequent Computational **S**ubgraph **M**ining
    - `RNN`
    - `SE` --> **S**hink and **E**xpand
    - `BC` --> **B**lock-wise **C**omponent
    - `ML` --> **M**odeling **L**anguage
    - `MM` --> **M**odularized **M**orphing
- **Accurarcy Computation**
    - `PP` --> **P**erformance **P**rediction
    - `ST` --> **ST**atistics derived from filter feature maps
    - `WP` --> **W**eight **P**rediction

### 2018
Neural Architecture Optimization
<br>
[[arXiv:1808.07233]](https://arxiv.org/abs/1808.07233)

Designing Adaptive Neural Networks for Energy-Constrained Image Classification
<br>
[[arXiv:1808.01550]](https://arxiv.org/abs/1808.01550)

Reinforced Evolutionary Neural Architecture Search
<br>
[[arXiv:1808.00193]](https://arxiv.org/abs/1808.00193)

MnasNet: Platform-Aware Neural Architecture Search for Mobile
<br>
[[arXiv:1807.11626]](https://arxiv.org/abs/1807.11626)

MaskConnect: Connectivity Learning by Gradient Descent
<br>
[[arXiv:1807.11473]](https://arxiv.org/abs/1807.11473)
[ECCV'18]

Efficient Neural Architecture Search with Network Morphism
<br>
[[arXiv:1806.10282]](https://arxiv.org/abs/1806.10282)
[[code]](https://github.com/jhfjhfj1/autokeras)

DARTS: Differentiable Architecture Search
:star:
<br>
[[arXiv:1806.09055]](https://arxiv.org/abs/1806.09055)
[[code]](https://github.com/quark0/darts)
--> `GD`

DPP-Net: Device-aware Progressive Search for Pareto-optimal Neural Architectures
<br>
[[arXiv:1806.08198]](https://arxiv.org/abs/1806.08198)
[[ICLR'18 Workshop]](https://openreview.net/forum?id=B1NT3TAIM)
--> `DE`

Path-Level Network Transformation for Efficient Architecture Search
<br>
[[arXiv:1806.02639]](https://arxiv.org/abs/1806.02639)
[[code]](https://github.com/han-cai/PathLevel-EAS)
[ICML'18]
--> `FT` `TP` `RL`

AlphaX: eXploring Neural Architectures with Deep Neural Networks and Monte Carlo Tree Search
<br>
[[arXiv:1805.07440]](https://arxiv.org/abs/1805.07440)
--> `SR` `MCTS` `PP`

Neural Architecture Construction using EnvelopeNets
<br>
[[arXiv:1803.06744]](https://arxiv.org/abs/1803.06744)
--> `ST`

Transfer Automatic Machine Learning
<br>
[[arXiv:1803.02780]](https://arxiv.org/abs/1803.02780)
--> `RL` `TL`

Neural Architecture Search with Bayesian Optimisation and Optimal Transport
<br>
[[arXiv:1802.07191]](https://arxiv.org/abs/1802.07191)
--> `BO`

Efficient Neural Architecture Search via Parameter Sharing
:star:
<br>
[[arXiv:1802.03268]](https://arxiv.org/abs/1802.03268)
[[code]](https://github.com/melodyguan/enas)
[ICML'18]
--> `PS` `RL` `SG`

Regularized Evolution for Image Classifier Architecture Search
<br>
[[arXiv:1802.01548]](https://arxiv.org/abs/1802.01548)
--> `EA`

GitGraph - from Computational Subgraphs to Smaller Architecture Search Spaces
<br>
[[arXiv:1801.05159]](https://arxiv.org/abs/1801.05159)
[[ICLR'18 Workshop]](https://openreview.net/forum?id=rkiO1_1Pz)
--> `SM`

A Flexible Approach to Automated RNN Architecture Generation
<br>
[[arXiv:1712.07316]](https://arxiv.org/abs/1712.07316)
[[ICLR'18 Workshop]](https://openreview.net/forum?id=SkOb1Fl0Z)
--> `RNN`

Peephole: Predicting Network Performance Before Training
:star:
<br>
[[arXiv:1712.03351]](https://arxiv.org/abs/1712.03351)
--> `PP`

Progressive Neural Architecture Search
<br>
[[arXiv:1712.00559]](https://arxiv.org/abs/1712.00559)
--> `SMBO`

MorphNet: Fast & Simple Resource-Constrained Structure Learning of Deep Networks
:star:
<br>
[[arXiv:1711.06798]](https://arxiv.org/abs/1711.06798)
--> `SE`

Simple And Efficient Architecture Search for Convolutional Neural Networks
:star:
<br>
[[arXiv:1711.04528]](https://arxiv.org/abs/1711.04528)
[[ICLR'18 Workshop]](https://openreview.net/forum?id=SySaJ0xCZ)
--> `NM`

Hierarchical Representations for Efficient Architecture Search
<br>
[[arXiv:1711.00436]](https://arxiv.org/abs/1711.00436)
[[ICLR'18]](https://openreview.net/forum?id=BJQRKzbA-)
--> `EA` `TP`

Practical Block-wise Neural Network Architecture Generation
<br>
[[arXiv:1708.05552]](https://arxiv.org/abs/1708.05552)
[CVPR'18]
--> `BC`

SMASH: One-Shot Model Architecture Search through HyperNetworks
<br>
[[arXiv:1708.05344]](https://arxiv.org/abs/1708.05344)
[[code]](https://github.com/ajbrock/SMASH)
[[ICLR'18]](https://openreview.net/forum?id=rydeCEhs-)
--> `WP`

Learning Transferable Architectures for Scalable Image Recognition
<br>
[[arXiv:1707.07012]](https://arxiv.org/abs/1707.07012)
--> `BC`

Efficient Architecture Search by Network Transformation
<br>
[[arXiv:1707.04873]](https://arxiv.org/abs/1707.04873)
[[code]](https://github.com/han-cai/EAS)
[AAAI'18]
--> `FT` `RL` `SE`

Learning Time/Memory-Efficient Deep Architectures with Budgeted Super Networks
:star:
<br>
[[arXiv:1706.00046]](https://arxiv.org/abs/1706.00046)
[[code]](https://github.com/TomVeniat/bsn)
[CVPR'18]
--> `DE` `GD`

Accelerating Neural Architecture Search using Performance Prediction
<br>
[[arXiv:1705.10823]](https://arxiv.org/abs/1705.10823)
[[ICLR'18 Workshop]](https://openreview.net/forum?id=BJypUGZ0Z)
--> `PP`

Understanding and Simplifying One-Shot Architecture Search
:star:
<br>
[[ICML'18]](http://proceedings.mlr.press/v80/bender18a/bender18a.pdf)
--> `1S`

### 2017
DeepArchitect: Automatically Designing and Training Deep Architectures
<br>
[[arXiv:1704.08792]](https://arxiv.org/abs/1704.08792)
[[code]](https://github.com/negrinho/deep_architect)
--> `ML`

Genetic CNN
<br>
[[arXiv:1703.01513]](https://arxiv.org/abs/1703.01513)
[[code]](https://github.com/aqibsaeed/Genetic-CNN)
[ICCV'17]
--> `EA`

Modularized Morphing of Neural Networks
<br>
[[arXiv:1701.03281]](https://arxiv.org/abs/1701.03281)
[[ICLR'17 Workshop]](https://openreview.net/forum?id=BJRIA3Fgg)
--> `FT` `MM`

Large-Scale Evolution of Image Classifiers
<br>
[[arXiv:1703.01041]](https://arxiv.org/abs/1703.01041)
[[ICML'17]](http://proceedings.mlr.press/v70/real17a.html)
--> `EA`

Designing Neural Network Architectures using Reinforcement Learning
<br>
[[arXiv:1611.02167]](https://arxiv.org/abs/1611.02167)
[[code]](https://github.com/bowenbaker/metaqnn)
[[ICLR'17]](https://openreview.net/forum?id=S1c2cvqee)
--> `RL`

Learning Curve Prediction with Bayesian Neural Networks
<br>
[[ICLR'17]](https://openreview.net/forum?id=S11KBYclx)
--> `PP`

Neural Architecture Search with Reinforcement Learning
<br>
[[arXiv:1611.01578]](https://arxiv.org/abs/1611.01578)
[[code (3rd)]](https://github.com/dhruvramani/Neural-Architecture-Search-with-RL)
[[ICLR'17]](https://openreview.net/forum?id=r1Ue8Hcxg)
--> `RL`

### 2016
Convolutional Neural Fabrics
<br>
[[arXiv:1606.02492]](https://arxiv.org/abs/1606.02492)
[[code:Caffe]](https://github.com/shreyassaxena/convolutional-neural-fabrics)
[[code:PyTorch]](https://github.com/vabh/convolutional-neural-fabrics)
[NIPS'16]

Network Morphism
:star:
<br>
[[arXiv:1603.01670]](https://arxiv.org/abs/1603.01670)
[[ICML'16]](http://proceedings.mlr.press/v48/wei16.html)
--> `FT` `MM`

Net2Net: Accelerating Learning via Knowledge Transfer
:star:
<br>
[[arXiv:1511.05641]](https://arxiv.org/abs/1511.05641)
[ICLR'16]
--> `KT`

### ~ 2015
A Hypercube-Based Indirect Encoding for Evolving Large-Scale
Neural Networks
<br>
[[Artificial Life journal'09]](http://axon.cs.byu.edu/~dan/778/papers/NeuroEvolution/stanley3**.pdf)
[[code]](https://github.com/MisterTea/HyperNEAT)
--> `EA` `TP`

## Useful Link
1. https://www.ml4aad.org/automl/literature-on-neural-architecture-search/
