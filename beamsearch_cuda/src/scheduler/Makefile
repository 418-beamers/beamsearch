CXX      = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -march=native -ffast-math

BENCHMARK = benchmark.o

SRCS = benchmark.cpp scheduler.cpp mlp/mlp_decay_scheduler.cpp

all: $(BENCHMARK)

$(BENCHMARK): $(SRCS) scheduler.h mlp/mlp_decay_scheduler.h
	$(CXX) $(CXXFLAGS) -o $(BENCHMARK) $(SRCS)

run: $(BENCHMARK)
	./$(BENCHMARK)

clean:
	rm -f $(BENCHMARK)

.PHONY: all run clean
