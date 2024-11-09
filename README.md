# SNN-Verification
## Requirements
- numpy                     1.23.5
- nengo                     3.2.0
- onnx                      1.13.1
- onnx-tf                   1.10.0
- onnxruntime               1.16.3

### The *Benchmarks* folder contains 4 NN controller benchmarks
- Adaptive Cruise Controller (with 3 and 5 hidden layers)
- Linear Inverted Pendulum
- Single Pendulum
- Double Pendulum

The *Range Verification* folder has the code for runnining the range verification and tightening queries on each of these benchmarks (labelled with their respective names. The files labelled \<Controller Name\> - Random Simulations have the code for generating random simulations using the neural networks
