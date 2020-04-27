openfst_dir=/usr/local

.PHONY: GPUCTC GPUDEN PATHWEIGHT CTCCRF
all: CTCCRF
GPUCTC:
	mkdir -p gpu_ctc/build
	cd gpu_ctc/build && cmake ..
	cd gpu_ctc/build && make	
GPUDEN:
	mkdir -p gpu_den/build
	cd gpu_den/build && cmake -D openfst_dir:STRING=${openfst_dir} ..
	cd gpu_den/build && make
PATHWEIGHT:
	mkdir -p path_weight/build
	cd path_weight/build && cmake -D openfst_dir:STRING=${openfst_dir} ..
	cd path_weight/build && make
CTCCRF: GPUCTC GPUDEN PATHWEIGHT
	python3 setup.py install

