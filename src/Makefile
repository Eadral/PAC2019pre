TARGET=FYArray.exe

CXX=icpc -g -xHost -O3 -ipo -no-prec-div -fp-model fast=2 -qopenmp -ltbb -std=c++11 -qopt-report=4 -qopt-report-phase=vec

INCLUDE=../include

OBJ=FYStorage.o main.o

$(TARGET):$(OBJ)
	$(CXX) $^ -o $@ 
	cp $(TARGET) ../
	
	#cd ..
	#./FYArray.exe 
	#profmerge -prof_dir profdata
	
	#cd src
	#$(CXX) -prof-use=nomerge -prof-func-groups -prof-dir=/home/zyc/profdata $^ -o $@ 
	#cp $(TARGET) ../
	

%.o:%.cpp
	$(CXX) -I$(INCLUDE) -c $< -o $@ 

.PHONY:clean
clean:
	rm *.o -f
