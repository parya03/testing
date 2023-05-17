cmd_/home/p7810456/kernelmodtest/testmod.mod := printf '%s\n'   testmod.o | awk '!x[$$0]++ { print("/home/p7810456/kernelmodtest/"$$0) }' > /home/p7810456/kernelmodtest/testmod.mod
