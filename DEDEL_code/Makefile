# Makefile for compiling and linking a multi-file C++ project using LibTorch

# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++17 -I ~/libtorch/include -I ~/libtorch/include/torch/csrc/api/include

# Linker flags
LDFLAGS = -L ~/libtorch/lib -ltorch -ltorch_cpu -lc10 -Wl,-rpath,~/libtorch/lib

# Source files
SRCS = SRC/main.cpp SRC/instance.cpp SRC/solution.cpp SRC/solver.cpp

# Object files (replace .cpp with .o)
OBJS = $(SRCS:.cpp=.o)

# Target executable
TARGET = main

# Default target
all: $(TARGET)

# Linking the target executable
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

# Compiling source files to object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
