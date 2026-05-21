# Image Classification Using a Multi-Layer-Perceptron

## Implementation of the MLP
All implementation was done in Python using the PyTorch library.

In order to allow for ease of use when configuring the network, I implemented it using a Python class whose constructer accepts various hyper-parameters for the model. Such hyper-parameters are the size of the input that the network will accept, the number of hidden layers and the number of neurons in each hidden layer, the number of neurons in the output layer, the activation function for each hidden layer.

For each hidden layer I created a an instance of torch.nn.Linear where I specified the number of neurons in the layer.
After each Linear layer I passed the output through the specified activation function and passed the output from the final hidden layer to the output layer.
The activation function for the output layer can also be specified in the consturctor.


## Configuring the Network


## Training


## Testing


## Results from Experiments
