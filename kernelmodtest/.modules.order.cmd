cmd_/home/p7810456/kernelmodtest/modules.order := {   echo /home/p7810456/kernelmodtest/testmod.ko; :; } | awk '!x[$$0]++' - > /home/p7810456/kernelmodtest/modules.order
