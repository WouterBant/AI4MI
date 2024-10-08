red:=$(shell tput bold ; tput setaf 1)
green:=$(shell tput bold ; tput setaf 2)
yellow:=$(shell tput bold ; tput setaf 3)
blue:=$(shell tput bold ; tput setaf 4)
magenta:=$(shell tput bold ; tput setaf 5)
cyan:=$(shell tput bold ; tput setaf 6)
reset:=$(shell tput sgr0)


data/TOY:
	python src/gen_toy.py --dest $@ -n 10 10 -wh 256 256 -r 50

data/TOY2:
	rm -rf $@_tmp $@
	python src/gen_two_circles.py --dest $@_tmp -n 1000 100 -r 25 -wh 256 256
	mv $@_tmp $@


# Extraction and slicing for Segthor
# data/segthor_train: data/segthor_train.zip
# 	$(info $(yellow)unzip $<$(reset))
# 	sha256sum -c data/segthor_train.sha256
# 	unzip -q $<


data/segthor_train: 
	$(info $(yellow)unzip $<$(reset))
	sha256sum -c data/segthor_train.sha256
	unzip -q data/segthor_train.zip
	
	$(info $(yellow)unzip $<$(reset))
	sha256sum -c data/test.zip.sha256
	unzip -q data/test.zip -d data/segthor_train


data/SEGTHOR: data/segthor_train
	$(info $(green)python $(CFLAGS) src/slice_segthor.py$(reset))
	rm -rf $@_tmp $@
	python $(CFLAGS) src/slice_segthor.py --source_dir $^ --dest_dir $@_tmp \
		--shape 256 256 --retains 10
	mv $@_tmp $@

data/SEGTHOR_MANUAL_SPLIT: data/segthor_train
	$(info $(green)python $(CFLAGS) src/slice_segthor.py$(reset))
	rm -rf $@_tmp $@
	python $(CFLAGS) src/slice_segthor.py --source_dir $^ --dest_dir $@_tmp \
		--shape 256 256 --retains 12 --create_test --retains_test 4
	mv $@_tmp $@


data/segthor_testonly: data/test.zip
	$(info $(yellow)unzip $<$(reset))
	sha256sum -c data/test.zip.sha256
	unzip -q $< -d data/segthor_train


data/SEGTHOR_TESTONLY: data/test
	$(info $(green)python $(CFLAGS) src/slice_segthor.py$(reset))
	rm -rf $@_tmp $@
	python $(CFLAGS) src/slice_segthor.py --source_dir $^ --dest_dir $@_tmp \
		--shape 256 256 --retains 0 --retains_test 0
	mv $@_tmp $@
