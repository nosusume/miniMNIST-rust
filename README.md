miniMNIST-rust

This project implements a **minimal** neural network in rust for classifying handwritten digits from the MNIST dataset. The entire implementation is **~300** lines of code.

# Features

- Two layer neural network (input -> hidden -> output)
- ReLU activation function for the hidden layer
- Softmax activation function for the output layer
- Cross-entropy loss function
- Stochastic gradient descent(SGD) optimizer

# Performance

```
epoch: 0, loss: -6523.138, accuracy: 0.95133334
epoch: 1, loss: -3070.3296, accuracy: 0.9629167
epoch: 2, loss: -2173.3323, accuracy: 0.9683333
epoch: 3, loss: -1680.5537, accuracy: 0.97083336
epoch: 4, loss: -1329.4857, accuracy: 0.9719167
epoch: 5, loss: -1063.8682, accuracy: 0.97258335
```