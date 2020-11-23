# rodinia-hls
Private repository for hosting the Rodinia HLS version for HiAccel

## Download

```shell
git clone https://github.com/SFU-HiAccel/rodinia-hls.git
```

## Setup Requirements
1. **Evaluated hardware platforms:**
  + **Host OS**
    + 64-bit Ubuntu 16.04.6 LTS
  + **Cloud FPGA**
    + Xilinx Alveo U200 - DDR4-based FPGA
2. **Software tools:**
   + **HLS tool**
     + Vitis 2019.2
     + Xilinx Runtime(XRT) 2019.2

## Usage
Each kernel has baseline version and multiple optimizations. In each baseline or optimization folder:

+ Generate the design for specified Target and Device:

```shell
make all TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform>
```

+ Run application in emulation or hardware
```shell
make check TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform>
```

+ Use ``make clean`` to remove all generated non-hardware files. 
+ Use ``make cleanall`` to remove all generated files.



## Kernel Introduction

### Backpropgation

### CFD 

### Hotspot







