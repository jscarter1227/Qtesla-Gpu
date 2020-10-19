nvcc -dc -x cu src/tests/test_qtesla.c -o obj/test_qtesla.c.o -g -G
nvcc -dc -x cu src/tests/cpucycles.c -o obj/cpucycles.c.o -g -G
nvcc -dc src/random/random.cu -o obj/random.cu.o -g -G
nvcc -dc src/sha3/fipsGpu.cu -o obj/fipsGpu.cu.o -g -G
nvcc -dc src/sign.cu -o obj/sign.cu.o -g -G
nvcc -dc src/sample.cu -o obj/sample.cu.o -g -G
nvcc -dc src/poly.cu -o obj/poly.cu.o -g -G
nvcc -dc src/pack.cu -o obj/pack.cu.o -g -G
nvcc -dc src/gauss.cu -o obj/gauss.cu.o -g -G
nvcc -dc src/consts.cu -o obj/consts.cu.o -g -G
nvcc -arch=compute_60 -code=sm_60 -link -Xcompiler -L./lib_I -D _X86_ -lqtesla -lm -L/usr/lib/ -lssl -lcrypto obj/sign.cu.o obj/pack.cu.o obj/sample.cu.o obj/gauss.cu.o obj/poly.cu.o obj/consts.cu.o obj/fipsGpu.cu.o obj/random.cu.o obj/cpucycles.c.o obj/test_qtesla.c.o -o test_qtesla-I -lrt -g -G