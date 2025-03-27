# Compiler and flags
CC = gcc
CFLAGS = -Wall -fPIC -O3 -mavx -mfma -fopenmp -MMD -MP
LDFLAGS = -shared -fopenmp

# Folders
SRC_DIR = src/drone
BUILD_DIR = build
LIB_DIR = lib

# Source and object files
SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# Output shared TARGET file
TARGET = $(LIB_DIR)/dronelib.so

# Default target: Compile everything
all: $(TARGET)

# Rule to build the shared TARGET
$(TARGET): $(OBJS)
	@mkdir -p $(LIB_DIR)  # Ensure lib directory exists
	$(CC) $(LDFLAGS) -o $@ $^

# Rule to build object files in the build directory from source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(BUILD_DIR)  # Ensure build directory exists
	$(CC) $(CFLAGS) -c -o $@ $<

# Clean up build files and the shared library
clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR)