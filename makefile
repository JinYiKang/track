CC ?= gcc
CXX ?= g++
CFLAGS ?= -fPIC -O2 -g2 -Wall -fpermissive -std=c++11
AR ?= ar
# OBJCOPY ?= objcopy
INC = \
	-I/home/jin/Documents/eigen-3.3.9/Eigen/ 

OBJ_DIR = ./
OBJ = $(patsubst %.cpp, %.o, $(wildcard *.cpp))
STATIC_LIB = $(OBJ_DIR)/libkalman.a

$(STATIC_LIB):$(OBJ)
	$(AR) rcs $@ $^
	@rm -rf $(OBJ)

all:$(OBJ)

$(OBJ): %.o :%.cpp
	$(CXX) -c $(INC) $(CFLAGS) -o $@  $<

clean:
	@rm -rf $(STATIC_LIB)

