TARGET=FYArray.exe

CXX=icpc -g -O3 -xHost -qopenmp -ltbb -std=c++11 -qopt-report=4 -qopt-report-phase=vec

INCLUDE=../include

OBJ=FYStorage.o main.o

$(TARGET):$(OBJ)
	$(CXX) $^ -o $@ 
	cp $(TARGET) ../

%.o:%.cpp
	$(CXX) -I$(INCLUDE) -c $< -o $@ 

.PHONY:clean
clean:
	rm *.o -f
