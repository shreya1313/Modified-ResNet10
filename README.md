# Deep-Learning-Mini-Project


## Goal

##### Design a modified Residual Network (ResNet) architecture with the highest test accuracy on the CIFAR- 10 image classification dataset, under the constraint that the model has no more than 5 million parameters.

## Our Proposed Architecture

![Alt text](arch.png)

## Repository Structure

- `models/` 
  - `.py` files - contains all the model architecture we experimented on using grid search, hyperparameters were tuned, and configurations were employed.
- `besthyperparams.json` - Hyperparameters for the best model.
- `gridsearch.py` - python script to train all the models present in `./models`, with all combinations of specified hyperparameters. It trains 45 models in total. 5 resnet variations with 9 combinations of hyperparameters each.
- `main_job.sbatch` - slurm job for running `gridsearch.py` on clsuters.
- `main_output` - output after running the slurm job.
- `resnet10.ipynb` - contains code for the most accurate model. 
- `resnet_model_93.54.ckpt` - checkpoint of the most accurate model. 
- `arc.png` - model architecture

## How to run
- Run the cells of the file `resnet10.ipynb`

## Results

| Model Architecture  | # of Blocks | # of Out Channels | # of Parameters | Optimizer | Learning Rate | Scheduler | Test Accuracy |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| **ResNet-10** | **[2,1,1,1]**  | **[64, 128, 256, 512]** | **4.98M** | **SGD** | **0.001** | **OneCycleLR** | **93.54%** |
| ResNet-10  | [2,1,1,1]  | [64, 128, 256, 512]  | 4.98M | SGD | 0.1 | LR decay | 91.45% |
| ResNet-12 | [2,2,2] | [16, 32, 64] | 0.19M | SGD | 0.1 | LR decay | 89.03% |
| ResNet-14-4 | [2,2,2,1] | [16, 32, 64, 512] | 3.15M | Adam | 0.01 | LR decay | 89.74% |
| ResNet-14-5 | [2,3,3,1,1] | [16, 32, 64, 128, 512] | 4.13M | SGD | 0.1 | LR decay | 89.45% |
| ResNet-18 | [2,2,2,2] | [64,128,256,512] | 11.17M | Adam | 0.001 | LR decay |92.98% |

## Documentation
- Project report can be found at [docs/project_report.pdf](<add link>)

## Authors
- Arushi Arora: aa10350@nyu.edu
- Saaketh Koundinya : sg7729@nyu.edu
- Shreya Agarwal : sa6981@nyu.edu
