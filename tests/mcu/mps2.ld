/* QEMU mps2-an385 / an386: code at 0x0 (4 MB), data in SRAM at 0x20000000 (4 MB).
 * .data is loaded in place by QEMU (newlib's semihosting crt0 does not copy it). */
MEMORY {
	FLASH (rx) : ORIGIN = 0x00000000, LENGTH = 4M
	RAM (rwx)  : ORIGIN = 0x20000000, LENGTH = 4M
}
ENTRY(_start)
SECTIONS {
	.text : { KEEP(*(.vectors)) *(.text*) *(.rodata*) KEEP(*(.init)) KEEP(*(.fini)) . = ALIGN(4); } > FLASH
	.ARM.exidx : { *(.ARM.exidx*) } > FLASH
	__etext = .;
	.data : { __data_start__ = .; *(.data*) . = ALIGN(4); __data_end__ = .; } > RAM
	.bss : { __bss_start__ = .; *(.bss*) *(COMMON) . = ALIGN(4); __bss_end__ = .; } > RAM
	end = .;
	__end__ = .;
	__stack = ORIGIN(RAM) + LENGTH(RAM);
	__HeapLimit = __stack - 0x10000;
}
