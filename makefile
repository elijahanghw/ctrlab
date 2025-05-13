# Compiler and flags
CC = gcc
NVCC = nvcc
CFLAGS = -Wall -fPIC -O3 -mavx -mfma -fopenmp -MMD -MP
NVCCFLAGS = -Xcompiler="-fPIC -fopenmp" -O3
LDFLAGS = -shared -fopenmp

# Folders
SRC_DIR_CPU = src/drone
SRC_DIR_GPU = src/drone_cuda
BUILD_DIR_CPU = build/cpu
BUILD_DIR_GPU = build/cuda
LIB_DIR = lib

# Source and object files and target
SRCS_CPU = $(wildcard $(SRC_DIR_CPU)/*.c)
OBJS_CPU = $(SRCS_CPU:$(SRC_DIR_CPU)/%.c=$(BUILD_DIR_CPU)/%.o)
TARGET_CPU = $(LIB_DIR)/dronelib.so

SRCS_GPU = $(wildcard $(SRC_DIR_GPU)/*.cu)
OBJS_GPU = $(SRCS_GPU:$(SRC_DIR_GPU)/%.cu=$(BUILD_DIR_GPU)/%.o)
TARGET_GPU = $(LIB_DIR)/dronelibcuda.so

# Default target: Compile everything
all: $(TARGET_CPU) $(TARGET_GPU)

# CPU Lib
$(TARGET_CPU): $(OBJS_CPU)
	@mkdir -p $(LIB_DIR)  # Ensure lib directory exists
	$(CC) $(LDFLAGS) -o $@ $^

# GPU Lib
$(TARGET_GPU): $(OBJS_GPU)
	@mkdir -p $(LIB_DIR)  # Ensure lib directory exists
	$(NVCC) -shared -Xcompiler="-fopenmp" -o $@ $^

# Compile CPU source files
$(BUILD_DIR_CPU)/%.o: $(SRC_DIR_CPU)/%.c
	@mkdir -p $(BUILD_DIR_CPU)  # Ensure build directory exists
	$(CC) $(CFLAGS) -c -o $@ $<

# Compile GPU source files
$(BUILD_DIR_GPU)/%.o: $(SRC_DIR_GPU)/%.cu
	@mkdir -p $(BUILD_DIR_GPU)  # Ensure build directory exists
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

# Clean up build files and the shared library
clean:
	rm -rf $(BUILD_DIR_CPU) $(BUILD_DIR_GPU) $(LIB_DIR)