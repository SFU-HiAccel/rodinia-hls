#!/bin/sh

# 
# v++(TM)
# runme.sh: a v++-generated Runs Script for UNIX
# Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
# 

if [ -z "$PATH" ]; then
  PATH=/local-scratch/Xilinx/Vitis/2020.1/bin:/local-scratch/Xilinx/Vitis/2020.1/bin
else
  PATH=/local-scratch/Xilinx/Vitis/2020.1/bin:/local-scratch/Xilinx/Vitis/2020.1/bin:$PATH
fi
export PATH

if [ -z "$LD_LIBRARY_PATH" ]; then
  LD_LIBRARY_PATH=
else
  LD_LIBRARY_PATH=:$LD_LIBRARY_PATH
fi
export LD_LIBRARY_PATH

HD_PWD='/localhdd/xingyut/rodinia-hls/Benchmarks/knn/knn_4_doublebuffer/_x.sw_emu.xilinx_u200_xdma_201830_2/knn/knn.sw_emu.xilinx_u200_xdma_201830_2/workload'
cd "$HD_PWD"

HD_LOG=runme.log
/bin/touch $HD_LOG

ISEStep="./ISEWrap.sh"
EAStep()
{
     $ISEStep $HD_LOG "$@" >> $HD_LOG 2>&1
     if [ $? -ne 0 ]
     then
         exit
     fi
}

EAStep vitis_hls -f workload.tcl -messageDb vitis_hls.pb
