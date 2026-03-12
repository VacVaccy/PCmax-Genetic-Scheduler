# Task Scheduler: $P||C_{max}$ Genetic Algorithm

![CUDA-Enabled](https://img.shields.io/badge/CUDA-Enabled-green.svg?style=flat-square&logo=nvidia)
![Status-Completed](https://img.shields.io/badge/Status-Completed-success.svg?style=flat-square)
![License-MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)

An implementation of a **Genetic Algorithm (GA)** designed to solve the $P||C_{max}$ scheduling problem (identically parallel machines). This project provides a rigorous comparison between three architectural paradigms: **Sequential**, **OpenMP (Multi-core CPU)**, and **CUDA (GPU acceleration)**.

---

## Problem Overview

The $P||C_{max}$ problem involves assigning $n$ independent tasks to $m$ identical machines. The objective is to minimize the **makespan** ($C_{max}$), which is the total time elapsing from the start of the first task to the completion of the last task.

The objective function is defined as:

$$C_{max} = \min_{S} \left( \max_{1 \le j \le m} \sum_{i \in J_j} p_i \right)$$

Where:
* $p_i$: Duration of task $i$.
* $J_j$: The set of tasks assigned to machine $j$.
* $S$: The set of all possible schedules.



---

## Implementation Highlights

* **Representation:** Individuals are represented as a vector of genes where the index is the task ID and the value is the assigned machine ID.
* **Initial Population:** A hybrid approach utilizing **Greedy Heuristics** (Shortest Queue First / LPT) to seed the population alongside random initialization for diversity.
* **Adaptive Mutation:** Implements a "High-Pressure" mutation operator. Instead of random swaps, it identifies tasks on the bottleneck machine (the one determining the current $C_{max}$) and reassigns them to underutilized machines.
* **Parallel Strategies:**
    * **OpenMP:** Employs thread-safe local populations with periodic migration/synchronization of the "Global Best" to prevent premature convergence.
    * **CUDA:** Massively parallel kernels for fitness evaluation and genetic operators using `curand`.

---

## Compilation & Execution

### Prerequisites
* `g++` (GCC 7+ recommended)
* `nvcc` (NVIDIA CUDA Toolkit)
* `OpenMP` library

### Build and Run Commands


##### Sequential version
```bash
g++ -std=c++17 main.cpp -o scheduler
```
```
./scheduler.exe <dataFile>
```

##### OpenMP version
```bash
g++ -std=c++17 -fopenmp main.cpp -o scheduler_omp
```
```
./scheduler.exe <dataFile>
```

##### CUDA version
```bash
nvcc -std=c++17 main.cu -o scheduler_cuda
```
```
scheduler.exe <dataFile>
```

### Configuration

The algorithm's hyper-parameters are fine-tuned via a config.json file. This allows for rapid experimentation without recompilation.
```json
{
  "mutationProbability": 0.35,
  "populationSize": 250,
  "chromosomesPreservedPercentage": 5,
  "splitPointRatio": 0.5,
  "generations": 500000,
  "mutationPressure": 0.25,
  "maxTime": 30
}
```

---

## References
1. Goldberg, D.E., Genetic Algorithms in Search, Optimization and Machine Learning, 1989.
2. Pinedo, M., Scheduling: Theory, Algorithms, and Systems, 2016.
3. Graham, R. L., Bounds on Multiprocessing Timing Anomalies.
