/* Tracing and structured logging.
 *
 * Every event goes to two independent sinks:
 *   stderr  human-readable, when tg_verbose >= the event's level
 *   --log   JSON Lines, every event regardless of verbosity
 *
 * Levels: 0 error, 1 info (-v), 2 debug (-vv).
 *
 * Usage: tr_begin(level, stage); tr_num/tr_str fields; tr_end(fmt, ...)
 * where fmt is the human-readable message (also stored as "msg"). */
#include "tg.h"

#include <stdarg.h>
#include <string.h>
#include <time.h>

int tg_verbose;
static FILE *logf;
static char ev[4096];
static size_t evn;
static int evlvl;
static const char *evstage;
static int evquiet;

static const char *lvlname[] = { "error", "info", "debug" };

double tr_now_ms(void)
{
	struct timespec ts;
	timespec_get(&ts, TIME_UTC);
	return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

void tr_open_log(const char *path)
{
	logf = fopen(path, "a");
	if (!logf) die(NULL, 0, "cannot open log '%s'", path);
}

void tr_close_log(void)
{
	if (logf) fclose(logf);
	logf = NULL;
}

/* Debug events reach the log only under -vv; errors and info always do. */
int tr_on(int level) { return tg_verbose >= level || (logf != NULL && level <= 1); }

static void put(const char *s)
{
	size_t k = strlen(s);
	if (evn + k < sizeof ev) {
		memcpy(ev + evn, s, k);
		evn += k;
		ev[evn] = 0;
	}
}

static void put_json_str(const char *s)
{
	char b[8];
	put("\"");
	for (; *s; s++) {
		unsigned char c = (unsigned char)*s;
		if (c == '"' || c == '\\') { b[0] = '\\'; b[1] = (char)c; b[2] = 0; put(b); }
		else if (c == '\n') put("\\n");
		else if (c == '\t') put("\\t");
		else if (c < 0x20) { snprintf(b, sizeof b, "\\u%04x", c); put(b); }
		else { b[0] = (char)c; b[1] = 0; put(b); }
	}
	put("\"");
}

void tr_begin(int level, const char *stage)
{
	evlvl = level;
	evstage = stage;
	evquiet = 0;
	evn = 0;
	ev[0] = 0;
}

/* The current event goes to the log only (its text is already shown elsewhere). */
void tr_quiet(void) { evquiet = 1; }

void tr_str(const char *key, const char *val)
{
	put(",");
	put_json_str(key);
	put(":");
	put_json_str(val ? val : "");
}

void tr_num(const char *key, double val)
{
	char b[64];
	put(",");
	put_json_str(key);
	if (val != val || val > 1e308 || val < -1e308) put(":null"); /* NaN/inf are not JSON */
	else {
		snprintf(b, sizeof b, ":%.9g", val);
		put(b);
	}
}

void tr_end(const char *fmt, ...)
{
	char msg[1024];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(msg, sizeof msg, fmt, ap);
	va_end(ap);
	if (!evquiet && tg_verbose >= evlvl && evlvl > 0) fprintf(stderr, "[%s] %s\n", evstage, msg);
	if (logf && (evlvl <= 1 || tg_verbose >= evlvl)) {
		char head[128];
		snprintf(head, sizeof head, "{\"t\":%.3f,\"level\":\"%s\",\"stage\":", tr_now_ms() / 1e3, lvlname[evlvl]);
		fputs(head, logf);
		char tail[sizeof ev];
		memcpy(tail, ev, evn + 1);
		evn = 0;
		ev[0] = 0;
		put_json_str(evstage);
		put(",\"msg\":");
		put_json_str(msg);
		fputs(ev, logf);
		fputs(tail, logf);
		fputs("}\n", logf);
		fflush(logf);
	}
}

/* Called by die() for user-facing (non-trapped) errors. */
void tr_error(const char *file, int line, const char *msg)
{
	if (!logf) return;
	tr_begin(0, "error");
	if (file && *file) {
		tr_str("file", file);
		tr_num("line", line);
	}
	tr_end("%s", msg);
}
