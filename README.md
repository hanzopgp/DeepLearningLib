# DeepLearningLib

## Presentation

Librairies for deep learning (MLP/CNN/Auto-Encoder etc...).

## Project architecture

<pre><code>DeepLearningLib/
      ├── src/                   
      |    ├── activation_functions/  (Folder containing activation functions)
      |    |      ├── LeakyRELU.py
      |    |      ├── Lin.py
      |    |      ├── ReLU.py
      |    |      ├── Sigmoid.py
      |    |      ├── Softmax.py
      |    |      └── Tanh.py
      |    ├── data/                  (Folder containing data generation functions)
      |    |      └── DataGeneration.py
      |    ├── layers/                (Folder contraining linear, conv1d... layers)
      |    |      ├── Linear.py
      |    |      ├── Conv1D.py
      |    |      ├── MaxPool.py
      |    |      └── Flatten.py
      |    ├── loss_functions/        (Folder containing loss functions)
      |    |      ├── BinaryCrossEntropy.py
      |    |      ├── CategoricalCrossEntropy.py
      |    |      ├── MeanAbsoluteError.py
      |    |      ├── MeanSquaredError.py
      |    |      ├── RootMeanSquaredError.py
      |    |      ├── SparseBinaryCrossEntropy.py
      |    |      ├── SparseCategoricalCrossEntropy.py
      |    |      └── SparseCategoricalCrossEntropySoftmax.py
      |    ├── network/               (Folder containing sequential module)
      |    |      └── Sequential.py
      |    ├── optimizer_functions/   (Folder containing optimizer functions such as SGD)
      |    |      ├── GradientDescent.py
      |    |      ├── MinibatchGradientDescent.py
      |    |      └── StochasticGradientDescent.py
      |    ├── utils/                 (Folder containing utility functions such as split_data())
      |    |      └── utils.py
      |    ├── Core.py                (File containing abstract classes such as Modules/Loss/Optimizer)
      |    ├── experiences.py         (File containing the MLP/AE/CNN experiences)
      |    ├── global_imports.py      (File containing the imports to avoid redundancy)
      |    ├── global_variables.py    (File containing all the global static variables)
      |    ├── main.py                (Main file containing the demo)
      |    └── unit_test.py           (Test file containing unit tests on easy problems) 
      ├── README.md		          
      └── LICENSE  
</pre></code>

## Implementation not seen in Project architecture

- Xavier initialization
- L2 regularisation

## Experiences

All those experiments were done on MNIST digits and fashion datasets :
- Multi layer perceptron image classification
- Autoencoder image reconstruction (with different latent space dimensions)
- Autoencoder removing noise (with different percentage of noise)
- Multi layer perceptron image classification with latent space representation (using different dimension)
