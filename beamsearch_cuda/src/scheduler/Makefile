# Compiler configuration
CXX      = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -march=native -ffast-math

# Targets
BENCHMARK = benchmark

# Source files
BENCH_SRC = benchmark.cpp

# Default target
all: $(BENCHMARK)

# Build benchmark
$(BENCHMARK): $(BENCH_SRC)
	$(CXX) $(CXXFLAGS) -o $(BENCHMARK) $(BENCH_SRC)

# Run the benchmark
run: $(BENCHMARK)
	./$(BENCHMARK)

# Clean up
clean:
	rm -f $(BENCHMARK)

.PHONY: all run clean
