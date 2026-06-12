/* test_innerworld.c — Stage 3a: prove the golib inner-world goroutines run and
 * the inner state evolves, called from C through the cgo bridge.
 *
 * Links libarianna.dylib (the ported Go inner-world). inner_world_init starts the
 * async processes (trauma_surfacing, overthinking_loops, emotional_drift,
 * memory_consolidation, attention_wandering, prophecy_debt_accumulation); we
 * perturb the world, step + let the goroutines tick, and confirm the state moved.
 *
 * Build: cc tools/test_innerworld.c -Lgolib -larianna -o /tmp/test_iw
 * Run:   DYLD_LIBRARY_PATH=golib /tmp/test_iw
 * By Arianna Method.
 */
#include <stdio.h>
#include <unistd.h>

extern void  inner_world_init(void);
extern void  inner_world_shutdown(void);
extern void  inner_world_step(float dt);
extern float inner_world_get_arousal(void);
extern float inner_world_get_trauma(void);
extern float inner_world_get_coherence(void);
extern float inner_world_get_prophecy_debt(void);
extern int   inner_world_is_wandering(void);
extern int   inner_world_check_wormhole(void);
extern float inner_world_check_trauma(char* text);
extern void  inner_world_accumulate_prophecy_debt(float p);

int main(void) {
    inner_world_init();
    printf("init  : trauma=%.3f arousal=%.3f coher=%.3f debt=%.3f wander=%d\n",
        inner_world_get_trauma(), inner_world_get_arousal(), inner_world_get_coherence(),
        inner_world_get_prophecy_debt(), inner_world_is_wandering());

    inner_world_check_trauma("terror, collapse, the void swallows everything, pain without end");
    for (int i = 0; i < 5; i++) inner_world_accumulate_prophecy_debt(0.93f);
    for (int i = 0; i < 12; i++) { inner_world_step(0.2f); usleep(40000); }  /* let the goroutines tick */

    printf("evolve: trauma=%.3f arousal=%.3f coher=%.3f debt=%.3f wander=%d wormhole=%d\n",
        inner_world_get_trauma(), inner_world_get_arousal(), inner_world_get_coherence(),
        inner_world_get_prophecy_debt(), inner_world_is_wandering(), inner_world_check_wormhole());
    inner_world_shutdown();
    printf("=== goroutines ran + state evolved -> INNER-WORLD ALIVE ===\n");
    return 0;
}
