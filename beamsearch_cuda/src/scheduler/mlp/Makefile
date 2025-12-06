CXX      = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -march=native -ffast-math

TARGET   = mlp_decay_scheduler.o

SRC      = mlp_decay_scheduler.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	@if [ ! -f mlp_weights.bin ]; then \
		echo "Error: mlp_weights.bin not found. Run the Python script first."; \
	else \
		./$(TARGET); \
	fi

clean:
	rm -f $(TARGET)

.PHONY: all run clean
