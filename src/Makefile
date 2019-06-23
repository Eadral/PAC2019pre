TARGET=FYArray.exe

CXX=icpc -g -xHost -O3 -ipo -par-num-threads=14 -par-affinity=granularity=fine,proclist=[0,2,4,6,8,10,12,14,16,18,20,22,24,26],explicit \
	-no-prec-div -funroll-all-loops -unroll-aggressive -fp-model fast=2 -qopenmp -ltbb -std=c++11 -qopt-report=5 -qopt-report-phase=vec

INCLUDE=../include

OBJ=FYStorage.o main.o

$(TARGET):$(OBJ)
	$(CXX) $^ -o $@ 
	cp $(TARGET) ../
	#../env.sh
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
