# SNN-Verification
## Requirements:
- numpy                     1.23.5
- nengo                     3.2.0
- onnx                      1.13.1
- onnx-tf                   1.10.0
- onnxruntime               1.16.3

### The *benchmark* folder contains 4 NN controller benchmarks
- Adaptive Cruise Controller (with 3 and 5 hidden layers)
- Linear Inverted Pendulum
- Single Pendulum
- Double Pendulum

The *Range Verification* folder has the code for runninning the range verification and tightening queries on each of these benchmarks
