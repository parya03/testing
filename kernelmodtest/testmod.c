#include <linux/module.h>
#include <linux/kernel.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Me");
MODULE_DESCRIPTION("A simple hello world module");
MODULE_VERSION("0.01");

static int my_init(void) {
    printk(KERN_INFO "Module loaded\n");
    return 0;
}

static void my_exit(void) {
    return;
}

module_init(my_init);
module_exit(my_exit);