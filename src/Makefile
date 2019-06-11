TARGET=FYArray.exe

CXX=icpc

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
