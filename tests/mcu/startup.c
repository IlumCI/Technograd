/* Minimal Cortex-M startup for the QEMU mps2 boards: a vector table whose
 * reset entry is newlib's semihosting crt0 (_start), which sets up the C
 * runtime, fetches argv from the host and calls main. */
extern void _start(void);
extern unsigned long __stack;
static void hang(void) { for (;;) {} }
__attribute__((section(".vectors"), used)) static void (*const vectors[16])(void) = {
	(void (*)(void))&__stack, _start, hang, hang, hang, hang, hang, 0, 0, 0, 0, hang, hang, 0, hang, hang,
};
