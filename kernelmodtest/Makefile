# obj-m    := testmod.o

# KDIR    := /lib/modules/$(shell uname -r)/build
# PWD    := $(shell pwd)

# default:
# 	$(MAKE) -C $(KDIR) SUBDIRS=$(shell pwd) modules

obj-m += testmod.o

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean