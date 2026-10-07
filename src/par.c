/* Shared-memory parallelism: a persistent thread pool with a dynamic
 * parallel-for.
 *
 * par_for(n, grain, fn, ctx) runs fn(ctx, lo, hi, tid) over [0, n) in chunks
 * of `grain`, claimed from an atomic counter, so uneven work (rows whose think
 * loops converge at different steps) balances itself. The calling thread works
 * as tid 0. A par_for issued from inside a worker runs inline, so nested
 * parallelism (an intra-op matmul inside a batch row) never deadlocks or
 * oversubscribes.
 *
 * Callers must not allocate (the allocator is single-threaded), trace, or call
 * die() from fn. */
#define _DEFAULT_SOURCE /* sysconf(_SC_NPROCESSORS_ONLN) under -std=c11 */
#include "tg.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

static pthread_t *workers;
static int nworkers;             /* threads besides the caller */
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv_go = PTHREAD_COND_INITIALIZER;
static pthread_cond_t cv_done = PTHREAD_COND_INITIALIZER;
static unsigned long generation;
static int running;              /* workers still inside the current job */
static int stopping;

static struct {
	void (*fn)(void *, int, int, int);
	void *ctx;
	int n, grain;
	atomic_int next;
} job;

static _Thread_local int in_par; /* this thread is executing a par_for body */

static void run_chunks(int tid)
{
	in_par = 1;
	for (;;) {
		int lo = atomic_fetch_add(&job.next, job.grain);
		if (lo >= job.n) break;
		int hi = lo + job.grain < job.n ? lo + job.grain : job.n;
		job.fn(job.ctx, lo, hi, tid);
	}
	in_par = 0;
}

static void *worker(void *arg)
{
	int tid = (int)(intptr_t)arg;
	unsigned long seen = 0;
	for (;;) {
		pthread_mutex_lock(&mu);
		while (generation == seen && !stopping) pthread_cond_wait(&cv_go, &mu);
		if (stopping) {
			pthread_mutex_unlock(&mu);
			return NULL;
		}
		seen = generation;
		pthread_mutex_unlock(&mu);
		run_chunks(tid);
		pthread_mutex_lock(&mu);
		if (--running == 0) pthread_cond_signal(&cv_done);
		pthread_mutex_unlock(&mu);
	}
}

static void par_shutdown(void)
{
	pthread_mutex_lock(&mu);
	stopping = 1;
	pthread_cond_broadcast(&cv_go);
	pthread_mutex_unlock(&mu);
	for (int i = 0; i < nworkers; i++) pthread_join(workers[i], NULL);
	free(workers);
	workers = NULL;
	nworkers = 0;
}

int par_cpus(void)
{
	long c = sysconf(_SC_NPROCESSORS_ONLN);
	return c > 0 ? (int)c : 1;
}

void par_init(int threads)
{
	if (threads <= 0) threads = par_cpus();
	if (threads > 256) threads = 256;
	if (nworkers || threads == 1) return;
	workers = malloc((size_t)(threads - 1) * sizeof *workers);
	if (!workers) die(NULL, 0, "out of memory");
	for (int i = 0; i < threads - 1; i++) {
		if (pthread_create(&workers[i], NULL, worker, (void *)(intptr_t)(i + 1)) != 0) break;
		nworkers++;
	}
	atexit(par_shutdown);
}

int par_threads(void) { return nworkers + 1; }

void par_for(int n, int grain, void (*fn)(void *ctx, int lo, int hi, int tid), void *ctx)
{
	if (n <= 0) return;
	if (grain < 1) grain = 1;
	if (nworkers == 0 || in_par || n <= grain) {
		fn(ctx, 0, n, 0);
		return;
	}
	pthread_mutex_lock(&mu);
	job.fn = fn;
	job.ctx = ctx;
	job.n = n;
	job.grain = grain;
	atomic_store(&job.next, 0);
	running = nworkers;
	generation++;
	pthread_cond_broadcast(&cv_go);
	pthread_mutex_unlock(&mu);

	run_chunks(0);

	pthread_mutex_lock(&mu);
	while (running > 0) pthread_cond_wait(&cv_done, &mu);
	pthread_mutex_unlock(&mu);
}
