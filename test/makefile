PCL_LIB = /usr/lib/x86_64-linux-gnu/libpcl*.so
OPENCV_LIB = /usr/local/lib/libopencv*
VTK_LIB = /usr/lib/x86_64-linux-gnu/libvtk*.so

INC = -I/home/jin/Documents/eigen-3.3.9/Eigen \
	  -I/usr/local/include/opencv4/ 	\
	  -I/usr/include/pcl-1.8 			\
	  -I/usr/include/vtk-6.3            \
	  -I../


STATIC_LIB = ../libkalman.a

main: main.o $(STATIC_LIB)
	g++ -o main main.o -lboost_system $(PCL_LIB) $(OPENCV_LIB) $(VTK_LIB) $(STATIC_LIB)

main.o: main.cc
	g++ -c main.cc -O2 $(INC)

$(STATIC_LIB):
	$(MAKE) -C ../

.PHONY: clean
clean:
	rm main main.o