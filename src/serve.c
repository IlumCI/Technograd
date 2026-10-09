/* Connections to the outside: `tgc stream` and `tgc serve`.
 *
 * The model stays a pure, statically planned function; this host layer
 * moves rows between it and the operating system:
 *
 *   tgc stream MODEL [SRC] [-o DST]
 *       one row in, one row out, flushed per row. SRC and DST are `-`
 *       (stdin / stdout), a file, or tcp://HOST:PORT (a connection this
 *       process opens). With a tcp:// SRC and no -o, results go back over
 *       the same connection, so tgc can work as a client of a row server.
 *
 *   tgc serve MODEL --listen [HOST:]PORT
 *       a TCP server. A connection speaks either the row protocol (send a
 *       row, receive its output row; any number per connection) or HTTP/1.1:
 *         GET  /health  -> ok
 *         GET  /        -> the model signature as JSON
 *         POST /run     -> body rows in, output rows out (text/plain)
 *       HOST defaults to 127.0.0.1; use 0.0.0.0 to accept other machines.
 *       Stateless models serve connections in parallel (one thread each,
 *       private arenas); a model with `update`s runs one row at a time, in
 *       arrival order, so its state stays consistent.
 *
 * Rows use the batch format: all inputs' values in signature order,
 * separated by commas or spaces. An output row is comma-separated, with the
 * think step counts appended as in `batch`. A malformed row gets an
 * `error: ...` line and the connection stays usable.
 *
 * SIGINT/SIGTERM stop both commands cleanly: the current row finishes,
 * outputs are flushed, --save-state is written, and the exit status is
 * 128 + signal number. SIGPIPE is ignored (a peer that disconnects ends
 * only its own connection). */
#define _POSIX_C_SOURCE 200809L
#include "tg.h"

#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

/* Read by connection threads and written by the signal handler: a lock-free
 * C11 atomic is safe in both roles (volatile sig_atomic_t is not, across threads). */
static atomic_int stop_sig;
#define tg_stop atomic_load(&stop_sig)

static void on_signal(int sig) { atomic_store(&stop_sig, sig); }

void tg_signals(void)
{
	struct sigaction sa;
	memset(&sa, 0, sizeof sa);
	sa.sa_handler = on_signal; /* no SA_RESTART: blocking reads return EINTR */
	sigemptyset(&sa.sa_mask);
	sigaction(SIGINT, &sa, NULL);
	sigaction(SIGTERM, &sa, NULL);
	signal(SIGPIPE, SIG_IGN);
}

int tg_exit_status(void) { return tg_stop ? 128 + tg_stop : 0; }

int tg_stopping(void) { return tg_stop != 0; }

/* ---- rows ---------------------------------------------------------------- */

typedef struct {
	const Module *m;
	int per, n, nt, *off;
	pthread_mutex_t mu;
	int serialize; /* stateful model, or threaded matmuls sharing the pool */
} Runner;

typedef struct {
	float *in, *out, *arena;
	int *steps;
	const float **ptr;
} Work;

static void runner_init(Runner *R, const Module *m)
{
	memset(R, 0, sizeof *R);
	R->m = m;
	R->off = xmalloc((size_t)(m->ninputs ? m->ninputs : 1) * sizeof *R->off);
	for (int i = 0; i < m->ninputs; i++) {
		R->off[i] = R->per;
		R->per += shape_numel(&m->val[m->inputs[i]].sh);
	}
	R->n = shape_numel(&m->val[m->output].sh);
	R->nt = m->nthink ? m->nthink : 1;
	R->serialize = m->nupd > 0 || par_threads() > 1;
	pthread_mutex_init(&R->mu, NULL);
}

/* Per-thread buffers, from malloc: connection threads must not use the
 * scoped allocator. */
static Work work_new(const Runner *R)
{
	Work w;
	w.in = calloc((size_t)R->per + 1, sizeof(float));
	w.out = calloc((size_t)R->n, sizeof(float));
	w.arena = calloc((size_t)(R->m->arena ? R->m->arena : 1), sizeof(float));
	w.steps = calloc((size_t)R->nt, sizeof(int));
	w.ptr = calloc((size_t)(R->m->ninputs ? R->m->ninputs : 1), sizeof(float *));
	if (!w.in || !w.out || !w.arena || !w.steps || !w.ptr) {
		fputs("tgc: out of memory\n", stderr);
		exit(1);
	}
	return w;
}

static void work_free(Work *w)
{
	free(w->in);
	free(w->out);
	free(w->arena);
	free(w->steps);
	free(w->ptr);
}

/* Run one text row and write its output (or an error line) to o.
 * Returns 0 for a blank/comment line, 1 otherwise. */
static int run_line(Runner *R, Work *w, const char *line, FILE *o)
{
	const char *bad;
	int k = io_parse_row(line, NULL, 0, &bad);
	if (k == 0) return 0;
	if (k < 0) {
		fprintf(o, "error: bad number near '%.20s'\n", bad);
		return 1;
	}
	if (k != R->per) {
		fprintf(o, "error: row has %d values, model %s needs %d\n", k, R->m->name, R->per);
		return 1;
	}
	io_parse_row(line, w->in, R->per, &bad);
	for (int i = 0; i < R->m->ninputs; i++) w->ptr[i] = w->in + R->off[i];
	if (R->serialize) pthread_mutex_lock(&R->mu);
	vm_run_into(R->m, w->ptr, w->out, w->steps, w->arena, NULL);
	if (R->serialize) pthread_mutex_unlock(&R->mu);
	io_write_row(o, 0, w->out, R->n, w->steps, R->m->nthink);
	return 1;
}

/* ---- sockets --------------------------------------------------------------- */

static void split_hostport(const char *spec, char *host, size_t hn, char *port, size_t pn, const char *dflt_host)
{
	const char *c = strrchr(spec, ':');
	if (!c) {
		snprintf(host, hn, "%s", dflt_host);
		snprintf(port, pn, "%s", spec);
		return;
	}
	size_t l = (size_t)(c - spec);
	if (l >= hn) l = hn - 1;
	memcpy(host, spec, l);
	host[l] = 0;
	if (!*host) snprintf(host, hn, "%s", dflt_host);
	snprintf(port, pn, "%s", c + 1);
}

static int tcp_connect(const char *spec)
{
	char host[256], port[32];
	split_hostport(spec, host, sizeof host, port, sizeof port, "127.0.0.1");
	struct addrinfo hints, *res;
	memset(&hints, 0, sizeof hints);
	hints.ai_socktype = SOCK_STREAM;
	int rc = getaddrinfo(host, port, &hints, &res);
	if (rc) die(NULL, 0, "tcp://%s: %s", spec, gai_strerror(rc));
	int fd = -1;
	for (struct addrinfo *a = res; a && fd < 0; a = a->ai_next) {
		fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
		if (fd >= 0 && connect(fd, a->ai_addr, a->ai_addrlen)) {
			close(fd);
			fd = -1;
		}
	}
	freeaddrinfo(res);
	if (fd < 0) die(NULL, 0, "tcp://%s: cannot connect: %s", spec, strerror(errno));
	return fd;
}

static int tcp_listen(const char *spec)
{
	char host[256], port[32];
	split_hostport(spec, host, sizeof host, port, sizeof port, "127.0.0.1");
	struct addrinfo hints, *res;
	memset(&hints, 0, sizeof hints);
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_flags = AI_PASSIVE;
	int rc = getaddrinfo(host, port, &hints, &res);
	if (rc) die(NULL, 0, "--listen %s: %s", spec, gai_strerror(rc));
	int fd = -1, one = 1;
	for (struct addrinfo *a = res; a && fd < 0; a = a->ai_next) {
		fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
		if (fd < 0) continue;
		setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
		if (bind(fd, a->ai_addr, a->ai_addrlen) || listen(fd, 64)) {
			close(fd);
			fd = -1;
		}
	}
	freeaddrinfo(res);
	if (fd < 0) die(NULL, 0, "--listen %s: %s", spec, strerror(errno));
	return fd;
}

/* getline without the line ending, surviving receive timeouts while no
 * stop was requested. */
static ssize_t read_line(char **buf, size_t *cap, FILE *f)
{
	for (;;) {
		errno = 0;
		ssize_t r = getline(buf, cap, f);
		if (r >= 0) {
			while (r > 0 && ((*buf)[r - 1] == '\n' || (*buf)[r - 1] == '\r')) (*buf)[--r] = 0;
			return r;
		}
		if ((errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) && !tg_stop) {
			clearerr(f);
			continue;
		}
		return -1;
	}
}

/* ---- stream ---------------------------------------------------------------- */

static FILE *open_stream(const char *spec, const char *mode, int *sock)
{
	if (!spec || !strcmp(spec, "-")) return *mode == 'r' ? stdin : stdout;
	if (!strncmp(spec, "tcp://", 6)) {
		*sock = tcp_connect(spec + 6);
		FILE *f = fdopen(*mode == 'r' ? *sock : dup(*sock), mode);
		if (!f) die(NULL, 0, "tcp://%s: %s", spec + 6, strerror(errno));
		return f;
	}
	FILE *f = fopen(spec, mode);
	if (!f) die(NULL, 0, "cannot open '%s'", spec);
	return f;
}

int stream_main(const Module *m, const char *src, const char *dst)
{
	Runner R;
	runner_init(&R, m);
	Work w = work_new(&R);
	int sock = -1, osock = -1;
	FILE *in = open_stream(src, "r", &sock);
	FILE *out;
	if (!dst && sock >= 0) { /* reply over the same connection */
		out = fdopen(dup(sock), "w");
		if (!out) die(NULL, 0, "%s: %s", src, strerror(errno));
	} else {
		out = open_stream(dst, "w", &osock);
	}
	char *line = NULL;
	size_t cap = 0;
	long rows = 0;
	double t0 = tr_now_ms();
	while (!tg_stop && read_line(&line, &cap, in) >= 0) {
		rows += run_line(&R, &w, line, out);
		if (fflush(out)) break; /* peer gone */
	}
	free(line);
	if (in != stdin) fclose(in);
	if (out != stdout) fclose(out);
	else fflush(out);
	if (tr_on(1)) {
		tr_begin(1, "stream");
		tr_num("rows", (double)rows);
		tr_num("ms", tr_now_ms() - t0);
		tr_end("%ld row(s) streamed%s", rows, tg_stop ? " (stopped by signal)" : "");
	}
	work_free(&w);
	return tg_exit_status();
}

/* ---- serve ----------------------------------------------------------------- */

typedef struct {
	Runner *R;
	int fd;
} Conn;

static pthread_mutex_t active_mu = PTHREAD_MUTEX_INITIALIZER;
static int active;
static long served;

static void http_reply(FILE *o, int code, const char *reason, const char *type, const char *body, size_t len)
{
	fprintf(o, "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n", code, reason, type, len);
	fwrite(body, 1, len, o);
}

static long http(Runner *R, Work *w, FILE *in, FILE *o, const char *req)
{
	long rows = 0;
	char method[8] = "", path[256] = "";
	if (sscanf(req, "%7s %255s", method, path) != 2) {
		http_reply(o, 400, "Bad Request", "text/plain", "bad request\n", 12);
		return 0;
	}
	char *line = NULL;
	size_t cap = 0;
	long clen = 0;
	while (read_line(&line, &cap, in) > 0)
		if (!strncasecmp(line, "content-length:", 15)) clen = strtol(line + 15, NULL, 10);
	free(line);
	char *body = NULL;
	size_t blen = 0;
	FILE *mo = open_memstream(&body, &blen);
	if (!mo) return 0;
	int code = 200;
	const char *reason = "OK", *type = "text/plain";
	if (!strcmp(method, "GET") && !strcmp(path, "/health")) {
		fputs("ok\n", mo);
	} else if (!strcmp(method, "GET") && !strcmp(path, "/")) {
		const Module *m = R->m;
		char sh[64];
		type = "application/json";
		fprintf(mo, "{\"model\":\"%s\",\"inputs\":[", m->name);
		for (int i = 0; i < m->ninputs; i++) {
			shape_str(&m->val[m->inputs[i]].sh, sh, sizeof sh);
			fprintf(mo, "%s{\"name\":\"%s\",\"type\":\"%s\"}", i ? "," : "", m->val[m->inputs[i]].name, sh);
		}
		shape_str(&m->val[m->output].sh, sh, sizeof sh);
		fprintf(mo, "],\"output\":\"%s\",\"values_per_row\":%d,\"think_loops\":%d,\"stateful\":%s}\n", sh, R->per, m->nthink,
			m->nupd ? "true" : "false");
	} else if (!strcmp(method, "POST") && !strcmp(path, "/run")) {
		if (clen < 0 || clen > (64L << 20)) {
			code = 413;
			reason = "Payload Too Large";
			fputs("body too large\n", mo);
		} else {
			char *req_body = malloc((size_t)clen + 1);
			size_t got = req_body ? fread(req_body, 1, (size_t)clen, in) : 0;
			if (req_body) {
				req_body[got] = 0;
				for (char *s = req_body; *s;) {
					char *nl = strchr(s, '\n');
					if (nl) *nl = 0;
					rows += run_line(R, w, s, mo);
					if (!nl) break;
					s = nl + 1;
				}
				free(req_body);
			}
		}
	} else {
		code = 404;
		reason = "Not Found";
		fputs("not found: GET /health, GET /, POST /run\n", mo);
	}
	fclose(mo);
	http_reply(o, code, reason, type, body, blen);
	free(body);
	return rows;
}

static void *conn_main(void *p)
{
	Conn c = *(Conn *)p;
	free(p);
	struct timeval tv = { 1, 0 }; /* wake once a second to notice a stop */
	setsockopt(c.fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
	FILE *in = fdopen(c.fd, "r"), *out = fdopen(dup(c.fd), "w");
	Work w = work_new(c.R);
	char *line = NULL;
	size_t cap = 0;
	long rows = 0;
	if (in && out && read_line(&line, &cap, in) >= 0) {
		if (!strncmp(line, "GET ", 4) || !strncmp(line, "POST ", 5)) {
			rows += http(c.R, &w, in, out, line);
		} else {
			do {
				rows += run_line(c.R, &w, line, out);
				if (fflush(out)) break;
			} while (!tg_stop && read_line(&line, &cap, in) >= 0);
		}
	}
	free(line);
	work_free(&w);
	if (out) fclose(out);
	if (in) fclose(in);
	else close(c.fd);
	pthread_mutex_lock(&active_mu);
	active--;
	served += rows;
	pthread_mutex_unlock(&active_mu);
	return NULL;
}

int serve_main(const Module *m, const char *listen_spec)
{
	Runner R;
	runner_init(&R, m);
	int lfd = tcp_listen(listen_spec);
	fprintf(stderr, "serve: model %s on %s (%d values per row%s); stop with SIGINT or SIGTERM\n", m->name, listen_spec, R.per,
		m->nupd ? ", stateful: rows run one at a time" : "");
	long conns = 0;
	sigset_t block, old;
	sigemptyset(&block);
	sigaddset(&block, SIGINT);
	sigaddset(&block, SIGTERM);
	while (!tg_stop) {
		struct pollfd pf = { lfd, POLLIN, 0 };
		int r = poll(&pf, 1, 500);
		if (r <= 0) continue;
		int fd = accept(lfd, NULL, NULL);
		if (fd < 0) continue;
		Conn *c = malloc(sizeof *c);
		if (!c) {
			close(fd);
			continue;
		}
		c->R = &R;
		c->fd = fd;
		pthread_mutex_lock(&active_mu);
		active++;
		pthread_mutex_unlock(&active_mu);
		pthread_t th;
		pthread_attr_t at;
		pthread_attr_init(&at);
		pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
		pthread_sigmask(SIG_BLOCK, &block, &old); /* signals go to this thread */
		int rc = pthread_create(&th, &at, conn_main, c);
		pthread_sigmask(SIG_SETMASK, &old, NULL);
		pthread_attr_destroy(&at);
		if (rc) {
			close(fd);
			free(c);
			pthread_mutex_lock(&active_mu);
			active--;
			pthread_mutex_unlock(&active_mu);
		}
		conns++;
	}
	close(lfd);
	for (int i = 0; i < 50; i++) { /* let open connections finish their row */
		pthread_mutex_lock(&active_mu);
		int a = active;
		pthread_mutex_unlock(&active_mu);
		if (!a) break;
		struct timespec ts = { 0, 100000000 };
		nanosleep(&ts, NULL);
	}
	pthread_mutex_lock(&active_mu);
	fprintf(stderr, "serve: stopped by signal %d after %ld connection(s), %ld row(s)\n", (int)tg_stop, conns, served);
	pthread_mutex_unlock(&active_mu);
	return tg_exit_status();
}
