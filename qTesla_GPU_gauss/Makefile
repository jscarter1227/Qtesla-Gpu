NVCC = nvcc
CUDAFLAGS = -dc -g -G
LINK = -arch=compute_60 -code=sm_60 -link -Xcompiler -L./lib_I -D _X86_ -lm -L/usr/lib/ -lssl -lcrypto -lrt
#-lqtesla 
STANDARD = $(NVCC) $(CUDAFLAGS)
CFLAGS = -x cu
OBJS = obj/sign.cu.o obj/pack.cu.o obj/sample.cu.o obj/gauss.cu.o obj/poly.cu.o obj/consts.cu.o obj/fipsGpu.cu.o obj/random.cu.o obj/cpucycles.c.o obj/test_qtesla.c.o

test_qtesla-I: $(OBJS)
	$(NVCC) $(LINK) $(OBJS) -o test_qtesla-I

obj/test_qtesla.c.o: src/tests/test_qtesla.c
	$(STANDARD) $(CFLAGS) src/tests/test_qtesla.c -o obj/test_qtesla.c.o

obj/cpucycles.c.o: src/tests/cpucycles.c
	$(STANDARD) $(CFLAGS) src/tests/cpucycles.c -o obj/cpucycles.c.o

obj/random.cu.o: src/random/random.cu
	$(STANDARD) src/random/random.cu -o obj/random.cu.o

obj/fipsGpu.cu.o: src/sha3/fipsGpu.cu
	$(STANDARD) src/sha3/fipsGpu.cu -o obj/fipsGpu.cu.o

obj/sign.cu.o: src/sign.cu
	$(STANDARD) src/sign.cu -o obj/sign.cu.o

obj/sample.cu.o: src/sample.cu
	$(STANDARD) src/sample.cu -o obj/sample.cu.o

obj/poly.cu.o: src/poly.cu
	$(STANDARD) src/poly.cu -o obj/poly.cu.o

obj/pack.cu.o: src/pack.cu
	$(STANDARD) src/pack.cu -o obj/pack.cu.o

obj/gauss.cu.o: src/gauss.cu
	$(STANDARD) src/gauss.cu -o obj/gauss.cu.o

obj/consts.cu.o: src/consts.cu
	$(STANDARD) src/consts.cu -o obj/consts.cu.o

.PHONY: clean

clean:
	rm -rf obj/* test_qtesla-I
