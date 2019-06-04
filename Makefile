all:
	cd src&&make

.PHONY:clean test
clean:
	rm FYArray.exe -f
	rm src/*.o -f
	rm src/FYArray.exe -f
	rm -rf hpctoolkit-*
	rm FYArray.exe.hpcstruct

test:
	hpcrun -t -e CPUTIME@5000 -e CYCLES -e INSTRUCTIONS@4000000 -e CACHE-MISSES ./FYArray.exe
	hpcstruct ./FYArray.exe
	hpcprof -S FYArray.exe.hpcstruct -I ./+ hpctoolkit-FYArray.exe-measurements
