# Ood
Practical federated learning system using either multiprocessing or network communication

## Installation
Run
~~~shell
pip install -r requirements.txt && make && pip install .
~~~

## Usage
The library makes use of skorch and PyTorch MPI. Making use of the library involves creating a PyTorch MPI
program with a skorch model where in place of the learning process, functions from this library are called.
Data used by endpoints must be PyTorch Dataloader form.