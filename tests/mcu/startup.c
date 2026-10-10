/* Minimal Cortex-M startup for the QEMU mps2 boards: a vector table whose
 * reset entry enables the FPU when the build uses one (CPACR CP10/CP11 full
 * access) and then enters newlib's semihosting crt0 (_start), which sets up
 * the C runtime, fetches argv from the host and calls main. */
extern void _start(void);
extern unsigned long __stack;
static void hang(void) { for (;;) {} }
static void reset(void)
{
#ifdef __ARM_FP
	*(volatile unsigned long *)0xE000ED88UL |= 0xFUL << 20;
	__asm volatile("dsb\n\tisb" ::: "memory");
#endif
	_start();
}
__attribute__((section(".vectors"), used)) static void (*const vectors[16])(void) = {
	(void (*)(void))&__stack, reset, hang, hang, hang, hang, hang, 0, 0, 0, 0, hang, hang, 0, hang, hang,
};
