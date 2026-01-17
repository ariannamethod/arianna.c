/*
 * test_blood.c — Test Blood compiler (dynamic C compilation)
 *
 * Demonstrates runtime LoRA compilation using Blood
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include "inner_world.h"

int main() {
    printf("=== BLOOD COMPILER TEST ===\n\n");

    // Initialize inner world
    inner_world_init();
    printf("Inner World initialized\n\n");

    // Get Blood temp directory
    char* temp_dir = blood_get_temp_dir();
    printf("Blood temp dir: %s\n\n", temp_dir);
    free(temp_dir);

    // 1. Compile a LoRA adapter
    printf("1. Compiling LoRA adapter...\n");
    char* lora_path = blood_compile_lora("test_lora", 128, 128, 8);
    if (lora_path) {
        printf("   SUCCESS! LoRA compiled to: %s\n", lora_path);

        // Try to load it
        void* handle = dlopen(lora_path, RTLD_NOW);
        if (handle) {
            printf("   Loaded successfully!\n");

            // Check for functions
            void* init_fn = dlsym(handle, "test_lora_init");
            void* apply_fn = dlsym(handle, "test_lora_apply");

            if (init_fn && apply_fn) {
                printf("   Functions found: test_lora_init, test_lora_apply\n");
            }

            dlclose(handle);
        } else {
            printf("   Load failed: %s\n", dlerror());
        }

        free(lora_path);
    } else {
        printf("   FAILED to compile LoRA\n");
    }

    printf("\n");

    // 2. Compile an emotional kernel
    printf("2. Compiling emotional kernel...\n");
    char* emotion_path = blood_compile_emotion("joy", 0.8f, 0.6f);
    if (emotion_path) {
        printf("   SUCCESS! Emotion compiled to: %s\n", emotion_path);

        void* handle = dlopen(emotion_path, RTLD_NOW);
        if (handle) {
            printf("   Loaded successfully!\n");

            void* check_fn = dlsym(handle, "joy_check");
            void* respond_fn = dlsym(handle, "joy_respond");

            if (check_fn && respond_fn) {
                printf("   Functions found: joy_check, joy_respond\n");
            }

            dlclose(handle);
        }

        free(emotion_path);
    } else {
        printf("   FAILED to compile emotion\n");
    }

    printf("\n");

    // 3. Compile raw C code
    printf("3. Compiling raw C code...\n");
    const char* raw_code =
        "#include <stdio.h>\n"
        "void hello_blood(void) {\n"
        "    printf(\"Hello from Blood-compiled code!\\n\");\n"
        "}\n";

    char* raw_path = blood_compile_raw("hello", raw_code);
    if (raw_path) {
        printf("   SUCCESS! Raw code compiled to: %s\n", raw_path);

        void* handle = dlopen(raw_path, RTLD_NOW);
        if (handle) {
            printf("   Loaded successfully!\n");

            // Get and call the function
            void (*hello_fn)(void) = dlsym(handle, "hello_blood");
            if (hello_fn) {
                printf("   Calling hello_blood(): ");
                hello_fn();
            }

            dlclose(handle);
        }

        free(raw_path);
    } else {
        printf("   FAILED to compile raw code\n");
    }

    printf("\n=== BLOOD TEST COMPLETE ===\n");

    inner_world_shutdown();
    return 0;
}
