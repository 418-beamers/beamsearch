# Compiler configuration
CXX      = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -march=native -ffast-math

# Targets
BENCHMARK = benchmark.o

# Source files
SRCS = benchmark.cpp scheduler.cpp mlp/mlp_decay_scheduler.cpp

# Default target
all: $(BENCHMARK)

# Build benchmark
$(BENCHMARK): $(SRCS) scheduler.h mlp/mlp_decay_scheduler.h
	$(CXX) $(CXXFLAGS) -o $(BENCHMARK) $(SRCS)

# Run the benchmark
run: $(BENCHMARK)
	./$(BENCHMARK)

# Clean up
clean:
	rm -f $(BENCHMARK)

.PHONY: all run clean
