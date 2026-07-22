# specifying the compiler
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    CXX ?= clang++
else
    CXX ?= g++
endif

INCLUDES = $(shell python -m pybind11 --includes)

# FILE EXTENSION: Python expects a specific suffix for C++ extensions 
SUFFIX = $(shell python -m pybind11 --extension-suffix)

# O3 is optimization, Wall enables warnings, std=c++17 tells c++ version, shared makes a library
# fPIC allows it to be loaded anywhere in memory.
# '-undefined dynamic_lookup' is REQUIRED on Mac so clang doesn't panic over missing Python symbols.
ifeq ($(UNAME_S),Darwin)
    UNDEFINED_FLAG = -undefined dynamic_lookup
else
    UNDEFINED_FLAG =
endif
CXXFLAGS = -O3 -Wall -std=c++17 -shared -fPIC $(UNDEFINED_FLAG)

# variables
srcdir = src_cpp
buildir = build

# The name MUST match PYBIND11_MODULE(riskengine, m) exactly.
target = $(buildir)/riskengine$(SUFFIX)

source = $(srcdir)/simulation.cpp
header = $(srcdir)/simulation.h

# this runs when "make" is executed
all: $(target)

$(target): $(source) $(header)
	@mkdir -p $(buildir)
	@echo "Compiling Risk Engine..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(source) -o $(target)
	@echo "Compilation Successful!"

clean:
	rm -f $(buildir)/*.so