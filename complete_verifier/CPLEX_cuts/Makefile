# Change the path based on your installation folder.
 CPX_PATH = /opt/ibm/ILOG/CPLEX_Studio221/cplex/
# CPX_PATH = /home/shiqi/cplex/cplex/
#CPX_PATH = /home/linyi/cplex/cplex/

CC = g++
ARCH = x86-64_linux

cpx: 
	$(CC) -g -L $(CPX_PATH)/lib/$(ARCH)/static_pic/ -L lib/ -I $(CPX_PATH)/include/ilcplex/ -o get_cuts get_cuts.cpp -lm -lcplex -ldl -lpthread

