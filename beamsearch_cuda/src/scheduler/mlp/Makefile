# Compiler configuration
CXX      = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -march=native -ffast-math

# Target executable name
TARGET   = mlp_decay_scheduler.o

# Source files
SRC      = mlp_decay_scheduler.cpp

# Default target (what runs when you type 'make')
all: $(TARGET)

# Link and Compile
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

# Helper to run the program (ensure weights exist first)
run: $(TARGET)
	@if [ ! -f mlp_weights.bin ]; then \
		echo "Error: mlp_weights.bin not found. Run the Python script first."; \
	else \
		./$(TARGET); \
	fi

# Clean up build files
clean:
	rm -f $(TARGET)

.PHONY: all run clean
