# rodinia-hls

Rodinia-hls is an FPGA version of the widely used GPU benchmark suite Rodinia, written in HLS (High-Level Synthesis) C/C++. This project was initiated by Dr. Zhenman Fang when he was a postdoc at UCLA, mentoring a few summer intern students. Now it is updated and maintained by the SFU-HiAccel group at SFU, led by Dr. Fang.

If you use Rodinia-hls in your research, please cite our FCCM 2018 paper:

> Jason Cong, Zhenman Fang, Michael Lo, Hanrui Wang, Jingxian Xu, Shaochong Zhang. "Understanding Performance Differences of FPGAs and GPUs". The 26th IEEE International Symposium on Field-Programmable Custom Computing Machines (FCCM 2018 short paper), Boulder CO, May 2018, pp. 172-175.

## Download

```shell
git clone https://github.com/SFU-HiAccel/rodinia-hls.git
```

## Setup Requirements

1. **Evaluated hardware platforms:**
  + **Host OS**
    + 64-bit Ubuntu 16.04.6 LTS
    + 64-bit Ubuntu 18.04.2 LTS
  + **Datacenter FPGA**
    + Xilinx Alveo U200 - DDR4-based FPGA
2. **Software tools:**
   + **HLS tool**
     + Vitis 2019.2
     + Xilinx Runtime(XRT) 2019.2

## Usage
Each kernel has folders for baseline version and multiple optimizations and a ``data`` folder.

In ``data`` folder, there are ``input.data`` as input data and ``check.data`` as reference data. 

In each baseline or optimization folder, you can run software emulation, hardware emulation and compile the kernel into FPGA hardware:

+ Generate the design for specified Target and Device:

```shell
make all TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform>
```

For FPGA platform names, they can be found in your install directory like /opt/xilinx/platforms/. For example, for U200 FPGA platform, use xilinx_u200_xdma_201830_2.

+ Run application in emulation or hardware
```shell
make check TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform>
```

+ Use ``make clean`` to remove all generated non-hardware files. 

+ Use ``make cleanall`` to remove all generated files. 

## Host Code

The host code is written in Xilinx OpenCL. We have abstracted the HLS common code and put it in Benchmarks/common folder. For each kernel's specific host code, i.e., OpenCL buffer allocation and transfer between CPU and FPGA, it's in the local_support.cpp file in each kernel's folder. So users only need to focus on local_support.cpp for the host code.

## Kernel Introduction

The kernels are written and optimized in Xilinx Vivado HLS C/C++. We follow a series of common optimizations, including tiling, pipeline, unrolling (parallelization), double buffering, and memory coalescing. The subfolders include the code of step-by-step HLS optimizations, which could be very useful for HLS beginners.

Below is a brief description of each kernel. You can find more details on [Rodinia Website](https://rodinia.cs.virginia.edu/doku.php).

### Backpropgation

Back Propagation is a machine-learning algorithm that trains the weights of connecting nodes on a layered neural network. 

### CFD 

The CFD solver is an unstructured grid finite volume solver for the three-dimensional Euler equations for compressible flow. 

### HotSpot

HotSpot is a widely used tool to estimate processor temperature based on an architectural floorplan and simulated power measurements.

### Kmeans

K-means is a clustering algorithm used extensively in data-mining and elsewhere, important primarily for its simplicity. Many data-mining algorithms show a high degree of data parallelism.

### KNN

NN (Nearest Neighbor) finds the k-nearest neighbors from an unstructured data set. The sequential NN algorithm reads in one record at a time, calculates the Euclidean distance from the target latitude and longitude, and evaluates the k nearest neighbors. 

### LavaMD

The code calculates particle potential and relocation due to mutual forces between particles within a large 3D space.

### Leukocyte

The leukocyte application detects and tracks rolling leukocytes (white blood cells) in in vivo video microscopy of blood vessels. The velocity of rolling leukocytes provides important information about the inflammation process, which aids biomedical researchers in the development of anti-inflammatory medications.

### LUD

LU Decomposition is an algorithm to calculate the solutions of a set of linear equations. The LUD kernel decomposes a matrix as the product of a lower triangular matrix and an upper triangular matrix.

### NW

Needleman-Wunsch is a nonlinear global optimization method for DNA sequence alignments.

### PathFinder

PathFinder uses dynamic programming to find a path on a 2-D grid from the bottom row to the top row with the smallest accumulated weights, where each step of the path moves straight ahead or diagonally ahead. 

### SRAD

SRAD (Speckle Reducing Anisotropic Diffusion) is a diffusion method for ultrasonic and radar imaging applications based on partial differential equations (PDEs).

### StreamCluster

For a stream of input points, it finds a predetermined number of medians so that each point is assigned to its nearest center. The quality of the clustering is measured by the sum of squared distances (SSQ) metric.

## Team Members and Contact:

Current Stuedents: [Alec Lu](http://www.sfu.ca/~fla30/), [Xingyu Tian](http://www.sfu.ca/~xingyut/)

Faculty: [Zhenman Fang](http://www.sfu.ca/~zhenman/group.html)

Past Contributors (at UCLA): Michael Lo (now PhD at UCLA), Hanrui Wang (now PhD at MIT), Jingxian Xu (now at Nvidia), Shaochong Zhang (now MASc at UCLA)

Special thanks to Falcon Computing to contribute the special memcpy APIs for memory coalescing optimizations.

If you have any questions, please feel free to contact Dr. Fang (zhenman@sfu.ca) or Xingyu (xingyut@sfu.ca)
