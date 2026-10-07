#include "unit.c" /* generated from examples/delta_memory.tg */
#include <stdio.h>
int main(void)
{
	const float e1[4] = { 1, 0, 0, 0 }, a[4] = { 1, 2, 3, 4 }, z[4] = { 0 };
	float out[4];
	tg_delta_memory_run(e1, a, out);                 /* write a */
	tg_delta_memory_run(e1, z, out);                 /* recall a (and erase) */
	int recalled = out[0] == 1 && out[3] == 4;
	float saved[TG_delta_memory_STATE_mem];          /* firmware-style save */
	tg_delta_memory_run(e1, a, out);                 /* write a again */
	for (int i = 0; i < TG_delta_memory_STATE_mem; i++) saved[i] = tg_delta_memory_state_mem[i];
	tg_delta_memory_reset();
	int wiped = 1;
	for (int i = 0; i < TG_delta_memory_STATE_mem; i++) wiped &= tg_delta_memory_state_mem[i] == 0;
	for (int i = 0; i < TG_delta_memory_STATE_mem; i++) tg_delta_memory_state_mem[i] = saved[i];  /* restore */
	tg_delta_memory_run(e1, z, out);
	int restored = out[0] == 1 && out[3] == 4;
	printf("recalled=%d reset_wiped=%d restored_after_save=%d\n", recalled, wiped, restored);
	return !(recalled && wiped && restored);
}
