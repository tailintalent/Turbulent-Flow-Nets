## Towards Physics-informed Deep Learning for Turbulent Flow Prediction
## Paper: 
Rui Wang, Karthik Kashinath, Mustafa Mustafa, Adrian Albert, Rose Yu [Towards Physics-informed Deep Learning for Turbulent Flow Prediction](https://ucsdml.github.io/jekyll/update/2020/08/23/TF-Net.html), KDD 2020

## [DataSet](https://drive.google.com/drive/folders/1VOtLjfAkCWJePiacoDxC-nrgCREKvrpE?usp=sharing.)
2000 velocity fields (![formula](https://render.githubusercontent.com/render/math?math=2000\times2\times256\times1792))
NOTE: According to the paper, the data used first splits up each raw image into seven 256x256 regions, then further downsamples them to 64x64 images. These downsampled images
are then fed into TF-net to be trained on. Currently, the raw data is ~7GB, which is difficult to load in all at once, so it may be helpful to break the data into chunks.
Current chunks:
- "rbc_data1.pt": 0 to 1300 of the 2000 raw images. Used as train set
- "rbc_data2.pt": 1300 to 1900 of the 2000 raw images. Used as val set
- "rbc_data3.pt": 1900 to 2000 of the 2000 raw images. Used as test set
Created by:
1. Open python shell
2. `import torch`
3. `full = torch.load("<path/to/rbc_data.pt>")`
4. `torch.save(full[0:1300, :,:,:].clone(), "rbc_data1.pt")`
5. `torch.save(full[1300:1900,:,:,:].clone(), "rbc_data2.pt")`
6. `torch.save(full[1900:2000,:,:,:].clone(), "rbc_data3.pt")`
Remember to have `.clone()`! Otherwise it will still save the full tensor.

### Abstract:
While deep learning has shown tremendous success in a wide range of domains, it remains a grand challenge to incorporate physical principles in a systematic manner to the design, training, and inference of such models. In this paper, we aim to predict turbulent flow by learning its highly nonlinear dynamics from spatiotemporal velocity fields of large-scale fluid flow simulations of relevance to turbulence modeling and climate modeling. We adopt a hybrid approach by marrying two well-established turbulent flow simulation techniques with deep learning. Specifically, we introduce trainable spectral filters in a coupled model of Reynolds-averaged Navier-Stokes (RANS) and Large Eddy Simulation (LES), followed by a specialized U-net for prediction. Our approach, which we call turbulent-Flow Net (TF-Net), is grounded in a principled physics model, yet offers the flexibility of learned representations. We compare our model, TF-Net, with state-of-the-art baselines and observe significant reductions in error for predictions 60 frames ahead. Most importantly, our method predicts physical fields that obey desirable physical characteristics, such as conservation of mass, whilst faithfully emulating the turbulent kinetic energy field and spectrum, which are critical for accurate prediction of turbulent flows.

### Model Architecture
<img src="./model.png" width="700" height="300">


### Velocity U & V Prediction and Ablation Study
![](Videos/all.gif)


## Description
1. Baselines/: Six baseline modules included in the paper.
2. TF-Net/: 
   1. model.py: TF-net pytorch implementation.
   1a. model_init.py: Modified version of model.py that currently has the dimensions to work with raw images
   2. penalty.py: a few regularizers we have tried.
   3. train.py: data loaders, train epoch, validation epoch, test epoch functions.
   3a. train_init.py: Same as train.py, but with modifications on data loading
   4. run_model.py: Scripts to train TF-Net
   4a. run_model_init.py: Same run_model.py, but with modifications to calling the dataset and saving model during training
   ```
   python run_model.py
   ```
3. Evaluation/:
   1. Evaluation.ipynb: contains the functions of four evaluation metrics.
   1a. Evaluation-Raw.ipynb: Uses the functions of Evaluation.ipynb to visualize the results of training on the raw 256x1796 images.
   2. radialProfile.py: a helper function for calculating energy spectrum.
4. Videos/: Videos of velocity u, v predictions and ablation study.

## Requirement 
* python 3.6
* pytorch 10.1
* matplotlib

## Cite
```
@article{Wang2020TF,
   title={Towards Physics-informed Deep Learning for Turbulent Flow Prediction},
   author={Rui Wang, Karthik Kashinath, Mustafa Mustafa, Adrian Albert, Rose Yu},
   journal={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining},
   Year = {2020}
}	
```
