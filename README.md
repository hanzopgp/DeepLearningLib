# DeepLearningLib

## Presentation

Librairies for deep learning (MLP/CNN/Auto-Encoder etc...).

## Project architecture

<pre><code>
DeepLearningLib/
      ├── src/                   
      |    ├── activation_functions/  (Folder containing activation functions)
      |    ├── data/                  (Folder containing data generation functions)
      |    ├── layers/                (Folder contraining linear, conv1d... layers)
      |    ├── loss_functions/        (Folder containing loss functions)
      |    ├── network/               (Folder containing sequential module)
      |    ├── optimizer_functions/   (Folder containing optimizer functions such as SGD)
      |    ├── utils/                 (Folder containing utility functions such as split_data())
      |    ├── Core.py                (Folder containing CNN/PyTorch tutorials)
      |    ├── experiences.py         (File containing the MLP/AE/CNN experiences)
      |    ├── global_imports.py      (File containing the imports to avoid redundancy)
      |    ├── global_variables.py    (File containing all the global static variables)
      |    ├── main.py                (Main file containing the demo)
      |    └── unit_test.py           (Test file containing unit tests on easy problems) 
      ├── README.md		          
      └── LICENSE  
</pre></code>
