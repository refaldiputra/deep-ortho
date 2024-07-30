# deep-ortho
Implementation of the paper "Deep Orthogonal Hypersphere Compression for Anomaly Detection", a spotlight in ICLR 2024.

By: Refaldi Intri Dwi Putra -  Department of Information and Communication Engineering, The University of Tokyo

Created with the purpose of the final report on Visual Media S/2024.

This repo is structured as         
    
    .
    ├── code      
    │    ├── confs          <- Hydra       
    │    ├── src            <- Program code
    │    ├── main.py        <- Main program
    │    └── analysis.ipynb <- Data and model analysis
    ├── README.md
    └── report.pdf          <- Report for this class
## How to run it?

We need to install:

`pip install torch torchvision hydra-core wandb`

To run it one can use the command in the terminal:

`cd code`

`python main.py`

Since we use Hydra, we can add the hyperparameters directly from the command line for example

`python main.py trainer.optimizer_enc.weight_decay=1e-2`

Folders `./outputs`, `./data` , and `./models` will be automatically created

You also need to log in to wandb in case you want to use it.

## What's in the report?

The report `report.pdf` contains a summary of the paper with more details on math but for simpler review can be found on this [Notion page](https://pumped-ring-6b0.notion.site/Implementation-of-Deep-Orthogonal-Hypersphere-Compression-for-Anomaly-Detection-ICLR-24-5ee46279f1d5410b88ee9267b4e48950?pvs=4).

## My implementation vs them?

This is the DOSHC result from the AUC mine vs them in the CIFAR-10 dataset.

Class | My Implementation | Their Results 
--- | --- | ---  
airplane | 0.78 | 0.80
automobile |0.80| 0.81
bird | 0.71 | 0.70 
cat |0.83 | 0.68 
deer |0.71 | 0.72
dog |0.65 | 0.72 
frog |0.62 | 0.83 
horse |0.63 | 0.74 
ship |0.77 | 0.83 
truck |0.77 | 0.81 

More details can be found on this wandb [report](https://api.wandb.ai/links/refaldiputra/xe285ujg).
