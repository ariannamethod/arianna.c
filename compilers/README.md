# Compilers — Dynamic Code Generation

Arianna's compiler infrastructure for runtime code generation.

## Blood (C)

Blood generates and compiles C code at runtime:
- LoRA adapters
- Emotional kernels
- Custom processing modules

Location: `inner_world/blood.go`

### Usage from C:
```c
// Compile a LoRA adapter
char* path = blood_compile_lora("my_lora", 128, 128, 8);
void* handle = dlopen(path, RTLD_NOW);

// Compile emotional kernel
char* emotion_path = blood_compile_emotion("joy", 0.8, 0.6);

// Compile raw C code
char* raw_path = blood_compile_raw("custom", c_code);
```

## High (Julia) — TODO

Mathematical brain for fast computations:
- Vectorized entropy
- Emotional weights
- N-gram processing

## H2O (Python) — Deferred

Python runtime for transformer scripts. Currently handled by separate wrapper.

## Philosophy

Like Linux kernel modules, Arianna can load/unload compiled code dynamically.
This enables self-modification and runtime adaptation.
