#!/bin/bash
if [ ! -n "$1"  ] ;then
    echo "Need TITLE!"
    exit
fi

# clean HERE
make clean
# clean END

make

stamp=`date +%s`
echo " " >> result
echo TITLE: $1 >> result
echo STAMP: $stamp >> result
date >> result

# test HERE
#hpcrun -t -e CPUTIME@5000 -e CYCLES -e INSTRUCTIONS@4000000 -e CACHE-MISSES ./FYArray.exe >> result
#hpcstruct ./FYArray.exe
#hpcprof -S FYArray.exe.hpcstruct -I ./+ hpctoolkit-FYArray.exe-measurements -o "./database-$stamp"
amplxe-cl -collect hotspots -r "r-$stamp-$1-hotspots" -search-dir . ./FYArray.exe >> result
amplxe-cl -collect uarch-exploration -r "r-$stamp-$1-uarch-exploration" -search-dir . ./FYArray.exe >> /dev/null
# test END

git add --all
git commit -m "AUTO:$1"
git push origin master
cat result

