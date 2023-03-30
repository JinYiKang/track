CC ?= gcc
CXX ?= g++
CFLAGS ?= -fPIC -g2 -Wall -fpermissive -std=c++11 -g -Ddebug
AR ?= ar
# OBJCOPY ?= objcopy
INC = \
	-I/home/jin/Documents/eigen-3.3.9/ \
	-I./header/

OBJ_DIR = ./
OBJ = $(patsubst %.cc, %.o, $(wildcard ./src/*.cc))
STATIC_LIB = $(OBJ_DIR)/libkalman.a

$(STATIC_LIB):$(OBJ)
	$(AR) rcs $@ $^
	@rm -rf $(OBJ)

all:$(OBJ)

$(OBJ): %.o :%.cc
	$(CXX) -c $(INC) $(CFLAGS) -o $@  $<

clean:
	@rm -rf $(STATIC_LIB)

