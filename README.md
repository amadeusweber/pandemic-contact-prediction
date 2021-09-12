# Simulating a pandempc spread using contact prediction

## About
Providing a pandimic simulation along with various contact prediction methods to test their effect on a diseases spread.

## Setup
Install [PyTorch](https://pytorch.org/)

Run
```
bash ./install.sh
```
To install the libraries ([pytorch_DGCNN](https://github.com/muhanzhang/pytorch_DGCNN), [SEAL](https://github.com/muhanzhang/SEAL)) and pyton packages specified in the [requirements.txt](requirements.txt).

If you're planning on using a virtual environment, make sure to activate it before running that line.

# Usage
The [run_sim.py](run_sim.py) script provides an example code for running the simulation using different prediction strategies.
The intermediate and final results will be stored in the sprecified direcory (e.g. [data/metrics/map_1_total/](data/metrics/map_1_total/)).
Intermediate results are marked by extending their name by '_pre'. They will be deleted when the final results are stored.
All results can then be visualized using the [run_visualization.py](run_visualization.py) script.
The plots and indicator values are saved in in the sprecified direcory (e.g. [data/plots/map_1_total/](data/plots/map_1_total/)).

