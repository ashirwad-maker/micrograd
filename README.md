# micrograd
![Micrograd](./micrograd.png)

In this repository, you'll find a compact Autograd engine, spiced up with a touch of fun! It's designed for backpropagation (reverse-mode autodiff) across a dynamically constructed Directed Acyclic Graph (DAG). On top of this, there's a lightweight neural networks library, boasting a PyTorch-like API. Each component is remarkably small, with roughly 100 lines of code for the DAG and 50 for the neural networks library. The DAG handles scalar values exclusively, dissecting each neuron into its fundamental arithmetic operations. Yet, this simplicity doesn't hinder its capability to construct comprehensive deep neural networks for binary classification tasks, as demonstrated in the accompanying notebook. So, buckle up and get ready for a wild ride through the realm of tiny yet mighty neural networks!
 ```python
 from micrograd.engine import Value
```