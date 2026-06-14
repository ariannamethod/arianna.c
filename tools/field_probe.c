/* field_probe.c — B/F-8 verification: prove the live shared field is genuinely
 * shared across PROCESSES. One invocation writes debt into the mmap'd field and
 * exits; another reads it back. Same value across two processes = MAP_SHARED works.
 *
 * Build: cc -Iariannamethod/core -Iariannamethod/notorch tools/field_probe.c \
 *           ariannamethod/core/libaml.a ariannamethod/notorch/libnotorch.a \
 *           -framework Accelerate -lm -o field_probe
 * Use:   ./field_probe <field> write <debt>   |   ./field_probe <field> read
 */
#include "ariannamethod.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <field> write <debt> | %s <field> read\n", argv[0], argv[0]); return 2; }
    am_init();
    if (am_field_attach(argv[1]) != 0) { fprintf(stderr, "[probe] attach '%s' failed\n", argv[1]); return 1; }
    if (!strcmp(argv[2], "write")) {
        float d = (argc > 3) ? (float)atof(argv[3]) : 7.5f;
        am_register_prophecy_debt(d);
        am_field_sync_out();
        printf("write: G.debt=%.5f  season=%d  → synced to %s\n", am_get_state()->debt, am_get_state()->season, argv[1]);
    } else {
        am_field_sync_in();
        const AM_State *s = am_get_state();
        printf("read:  G.debt=%.5f  dissonance=%.5f  season=%d  (from %s)\n", s->debt, s->dissonance, s->season, argv[1]);
    }
    am_field_detach();
    return 0;
}
