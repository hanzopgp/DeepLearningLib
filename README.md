# DeepLearningLib

## Presentation

Librairies for deep learning (MLP/CNN/Auto-Encoder etc...).

## Project architecture

<pre><code>DeepLearningLib/
      ├── nndiy/                   
      |    ├── __init__.py        (Contains the sequential object and global variables)
      |    ├── activation.py      (Contains the activation functions such as ReLU/Tanh...)
      |    ├── core.py            (Contains the abstract classes)
      |    ├── early_stopping.py  (Contains the early stopping objects)
      |    ├── layer.py           (Contains the layers object such as Linear/Dropout...)
      |    ├── loss.py            (Contains the loss objects such as MSE/BCE...)
      |    ├── optimizer.py       (Contains the optimizer objects such as SGD/ADAM...)
      |    └── utils.py           (Contains the utility functions such as min_max_scale/one_hot...)
      ├── cnn_demo.py             (Contains the demo for CNN)
      ├── experiences.py          (Contains the MLP/AE/CNN experiences)
      ├── mlp_unit_test.py        (Contains MLP unit tests on simple problems) 
      ├── report/                 (Folder containing the images and report)
      |    ├── img_report/
      |    └── report.pdf         
      ├── README.md		          
      └── LICENSE  
</pre></code>

## Features implemented

- Linear/Convo1D/MaxPool1D/AvgPool1D/Flatten/Dropout layers
- GD/MGD/SGD/ADAM optimizers
- LearkyReLU/ReLU/Identity/Tanh/Sigmoid/Softmax activation functions
- MAE/MSE/RMSE/BCE/SBCE/CCE/SCCE/SCCESoftmax loss functions
- Uniform/Xavier initialization
- L1/L2 regularisation
- EarlyStopping callback

## Experiences

All those experiments were done on MNIST digits and fashion datasets :
- Multi layer perceptron image classification
- Autoencoder image reconstruction (with different latent space dimensions)
- Autoencoder removing noise (with different percentage of noise)
- Multi layer perceptron image classification with latent space representation (using different dimension)
- SGD/ADAM/Tanh/ReLU benchmarks on MNIST
- 1D CNN on MNIST
