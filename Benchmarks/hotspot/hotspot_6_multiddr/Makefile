#include ../common.mk
.PHONY: help

help::
	$(ECHO) "Makefile Usage:"	
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform>"
	$(ECHO) "      Command to generate the design for specified Target and Device."
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""
	$(ECHO) "  make check TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform>"
	$(ECHO) "      Command to run application in emulation."
	$(ECHO) ""
	$(ECHO) "  make build TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform>"
	$(ECHO) "      Command to build xclbin application."
	$(ECHO) ""
	$(ECHO) "  make run_nimbix DEVICE=<FPGA platform>"
	$(ECHO) "      Command to run application on Nimbix Cloud."
	$(ECHO) ""
	$(ECHO) "  make aws_build DEVICE=<FPGA platform>"
	$(ECHO) "      Command to build AWS xclbin application on AWS Cloud."
	$(ECHO) ""

# Compiler tools
# XILINX_SDX ?= /local-scratch/SDx/SDx/2019.1
# XILINX_XRT ?= /opt/xilinx/xrt
# XILINX_SDK ?= $(XILINX_SDX)/../../SDK/2019.1
# XILINX_VIVADO ?= /local-scratch/SDx/Vivado/2019.1
# XILINX_VIVADO_HLS ?= $(XILINX_SDX)/Vivado_HLS.     

# XOCC_LINK_OPTS += --slr $(KERNAL_NAME)_1:SLR1 #SLR: specify which SLR to use
XOCC_LINK_OPTS += --connectivity.sp $(KERNAL_NAME)_1.m_axi_temp:DDR[0]
XOCC_LINK_OPTS += --connectivity.sp $(KERNAL_NAME)_1.m_axi_power:DDR[1]
XOCC_LINK_OPTS += --connectivity.sp $(KERNAL_NAME)_1.m_axi_result:DDR[2]	
	#specify all interfaces 
	#remove m_axi_ 
	
# Kernal Name
APP=hotspot
KERNAL_NAME=workload

# Points to Utility Directory
COMMON_REPO = ../../common
ABS_COMMON_REPO = $(shell readlink -f $(COMMON_REPO))

TARGETS := hw
TARGET := $(TARGETS)
DEVICE := $(DEVICES)
XCLBIN := ./xclbin

include ../utils.mk

#LDCLFLAGS += --profile_kernel data:all:all:all:all --profile_kernel stall:all:all:all:all

DSA := $(call device2dsa, $(DEVICE))
BUILD_DIR := ./_x.$(TARGET).$(DSA)

BUILD_DIR_KERNEL = $(BUILD_DIR)/$(APP)

#CXX := $(XILINX_SDX)/bin/xcpp
#XOCC := $(XILINX_SDX)/bin/xocc
CXX := $(XILINX_VITIS)/bin/xcpp
VPP := $(XILINX_VITIS)/bin/v++

#Include Libraries
include $(ABS_COMMON_REPO)/libs/opencl/opencl.mk
include $(ABS_COMMON_REPO)/libs/xcl2/xcl2.mk
CXXFLAGS += $(xcl2_CXXFLAGS)
LDFLAGS += $(xcl2_LDFLAGS)
HOST_SRCS += $(xcl2_SRCS)
CXXFLAGS += $(opencl_CXXFLAGS) -Wall -O0 -g -std=c++14 -I$(ABS_COMMON_REPO)
LDFLAGS += -lxilinxopencl -lpthread -lrt -lstdc++ -lmpfr -lgmp -lhlsmc++-GCC46 -lIp_floating_point_v7_0_bitacc_cmodel -lIp_xfft_v9_1_bitacc_cmodel -lIp_fir_compiler_v7_2_bitacc_cmodel -lIp_dds_compiler_v6_0_bitacc_cmodel -L$(XILINX_XRT)/lib/ -L$(XILINX_VIVADO)/lnx64/tools/fpo_v7_0 -L$(XILINX_VIVADO)/lnx64/lib/csim -L$(XILINX_VIVADO)/lnx64/tools/dds_v6_0 -L$(XILINX_VIVADO)/lnx64/tools/fir_v7_0 -L$(XILINX_VIVADO)/lnx64/tools/fft_v9_1 -Wl,-rpath-link,$(XILINX_XRT)/lib -Wl,-rpath,$(XILINX_VIVADO)/lnx64/lib/csim -Wl,-rpath,$(XILINX_VIVADO)/lnx64/tools/fpo_v7_0 -Wl,-rpath,$(XILINX_VIVADO)/lnx64/tools/fft_v9_1 -Wl,-rpath,$(XILINX_VIVADO)/lnx64/tools/fir_v7_0 -Wl,-rpath,$(XILINX_VIVADO)/lnx64/tools/dds_v6_0

HOST_SRCS += $(ABS_COMMON_REPO)/harness.c $(ABS_COMMON_REPO)/support.c src/local_support.cpp
HOST_HDRS += $(ABS_COMMON_REPO)/support.h 
HOST_ARGS = ../data/input.data ../data/check.data

# Host compiler global settings
CXXFLAGS += -fmessage-length=0 -DSDX_PLATFORM=$(DEVICE) -D__USE_XOPEN2K8 -I$(XILINX_XRT)/include/ -I$(XILINX_VIVADO)/include/
LDFLAGS += -lrt -lstdc++ 

# Kernel compiler global settings
CLFLAGS += -t $(TARGET) --platform $(DEVICE) --save-temps

EXECUTABLE = host
CMD_ARGS = $(XCLBIN)/$(APP).$(TARGET).$(DSA).xclbin

EMCONFIG_DIR = $(XCLBIN)/$(DSA)

BINARY_CONTAINERS += $(XCLBIN)/$(APP).$(TARGET).$(DSA).xclbin
BINARY_CONTAINER_vadd_OBJS += $(XCLBIN)/$(APP).$(TARGET).$(DSA).xo

CP = cp -rf

.PHONY: all clean cleanall docs emconfig
all: check-devices $(EXECUTABLE) $(BINARY_CONTAINERS) emconfig

.PHONY: exe
exe: $(EXECUTABLE)

.PHONY: build
build: $(BINARY_CONTAINERS)

# Building kernel
$(XCLBIN)/$(APP).$(TARGET).$(DSA).xo: src/$(APP).cpp
	mkdir -p $(XCLBIN)
	$(VPP) $(CLFLAGS) --temp_dir $(BUILD_DIR_KERNEL) -c -k $(KERNAL_NAME) -I'$(<D)' -o'$@' '$<'
	#$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_KERNEL) -c -k $(KERNAL_NAME) -I'$(<D)' -o'$@' '$<'
$(XCLBIN)/$(APP).$(TARGET).$(DSA).xclbin: $(BINARY_CONTAINER_vadd_OBJS)
	mkdir -p $(XCLBIN)
	$(VPP) $(CLFLAGS) --temp_dir $(BUILD_DIR_KERNEL) -R2 -l $(LDCLFLAGS) --connectivity.nk $(KERNAL_NAME):1 -j 8 -o'$@' $(+) $(XOCC_LINK_OPTS)
	#$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_KERNEL) -l $(LDCLFLAGS) --nk $(KERNAL_NAME):1 -o'$@' $(+) $(XOCC_LINK_OPTS)

# Building Host
$(EXECUTABLE): check-xrt $(HOST_SRCS) $(HOST_HDRS)
	$(CXX) $(CXXFLAGS) $(HOST_SRCS) $(HOST_HDRS) -o '$@' $(LDFLAGS)

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(DEVICE) --od $(EMCONFIG_DIR)

check: all
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	$(CP) $(EMCONFIG_DIR)/emconfig.json .
	XCL_EMULATION_MODE=$(TARGET) ./$(EXECUTABLE) $(HOST_ARGS) $(XCLBIN)/$(APP).$(TARGET).$(DSA).xclbin
else
	 ./$(EXECUTABLE) $(HOST_ARGS) $(XCLBIN)/$(APP).$(TARGET).$(DSA).xclbin
endif
#	sdx_analyze profile sdaccel_profile_summary.csv

run_nimbix: all
	$(COMMON_REPO)/utility/nimbix/run_nimbix.py $(EXECUTABLE) $(CMD_ARGS) $(DSA)

aws_build: check-aws_repo $(BINARY_CONTAINERS)
	$(COMMON_REPO)/utility/aws/run_aws.py $(BINARY_CONTAINERS)

# Cleaning stuff
clean:
	-$(RMDIR) $(EXECUTABLE) $(XCLBIN)/{*sw_emu*,*hw_emu*} 
	-$(RMDIR) profile_* TempConfig system_estimate.xtxt *.rpt *.csv 
	-$(RMDIR) src/*.ll _xocc_* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb

cleanall: clean
	-$(RMDIR) $(XCLBIN)
	-$(RMDIR) _x.* output.data *.run_summary .run


