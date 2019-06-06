#!/bin/bash
if [ ! -n "$1"  ] ;then
    echo "Need TITLE!"
    exit
fi

make clean
rm FYArray.exe.hpcstruct
make
hpcrun -t -e CPUTIME@5000 -e CYCLES -e INSTRUCTIONS@4000000 -e CACHE-MISSES ./FYArray.exe >> result

stamp=`date +%s`
echo TITLE: $1 >> result
echo STAMP: $stamp >> result
date >> result

hpcstruct ./FYArray.exe
hpcprof -S FYArray.exe.hpcstruct -I ./+ hpctoolkit-FYArray.exe-measurements -o "./database-$stamp"

git add --all
git commit -m "AUTO:$1"
git push origin master
cat result

