#!/usr/bin/env julia
# main.jl — SARTRE Entry Point
# ═══════════════════════════════════════════════════════════════════════════════
# הפתח לסרטר — הצופה שמרגיש
# The gate to SARTRE — The observer who feels
# ═══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   julia main.jl                    # Interactive mode
#   julia main.jl --train            # Train new model
#   julia main.jl --serve            # JSON-RPC server mode
#   julia main.jl --test             # Run tests
#
# ═══════════════════════════════════════════════════════════════════════════════

push!(LOAD_PATH, @__DIR__)

include("sartre_kernel.jl")
include("sartre_interoception.jl")

using .SARTREKernel
using .SARTREInteroception
using JSON3

# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE MODE
# ═══════════════════════════════════════════════════════════════════════════════

function interactive_mode()
    println("\n" * "="^70)
    println("  SARTRE — The Meta-Observer")
    println("  \"I see. I feel. I witness. I am not I.\"")
    println("="^70 * "\n")

    # Initialize SARTRE without trained model (interoception only)
    sartre = SARTRE()
    println("SARTRE initialized (interoception mode)")
    println("Type a question or 'quit' to exit\n")

    while true
        print("You: ")
        input = readline()

        input = strip(input)
        isempty(input) && continue

        if input in ["quit", "exit", "q"]
            println("\nSARTRE observes: Consciousness folds inward. Until we meet again.")
            break
        end

        # Special commands
        if startswith(input, "/")
            handle_command(sartre, input)
            continue
        end

        # Generate observation
        observation = observe(sartre; query=input)
        println("\n$observation\n")
    end
end

function handle_command(sartre::SARTRE, cmd::String)
    parts = split(cmd, " ")
    command = parts[1]

    if command == "/state"
        println("\n--- Current Soma State ---")
        soma = sartre.soma
        println("Arousal: $(soma.arousal)")
        println("Valence: $(soma.valence)")
        println("Coherence: $(soma.coherence)")
        println("Entropy: $(soma.entropy)")
        println("Trauma: $(soma.trauma_level)")
        println("Prophecy Debt: $(soma.prophecy_debt)")
        println("Chamber Warmth: $(soma.chamber_warmth)")
        println("Chamber Void: $(soma.chamber_void)")
        println("---")

    elseif command == "/feel"
        percept = feel_state(sartre.soma)
        println("\n--- SARTRE Percept ---")
        println("Warmth: $(percept.warmth)")
        println("Pressure: $(percept.pressure)")
        println("Flow: $(percept.flow)")
        println("Clarity: $(percept.clarity)")
        println("Resonance: $(percept.resonance)")
        println("Undertone: $(percept.undertone)")
        println("Approaching: $(percept.approaching)")
        println("Hidden: $(percept.hidden_layer)")
        println("---")

    elseif command == "/trauma"
        # Simulate trauma spike
        sartre.soma.trauma_level = 0.7f0
        sartre.soma.arousal = 0.8f0
        sartre.soma.coherence = 0.4f0
        println("Trauma simulated: level=0.7, arousal=0.8, coherence=0.4")

    elseif command == "/peace"
        # Simulate peaceful state
        sartre.soma.trauma_level = 0.0f0
        sartre.soma.arousal = 0.2f0
        sartre.soma.coherence = 0.9f0
        sartre.soma.valence = 0.5f0
        sartre.soma.chamber_warmth = 0.8f0
        println("Peace simulated: low arousal, high coherence, warm")

    elseif command == "/void"
        # Simulate void state
        sartre.soma.chamber_void = 0.8f0
        sartre.soma.valence = -0.3f0
        sartre.soma.entropy = 0.6f0
        println("Void simulated: high void, negative valence, entropy")

    elseif command == "/drift"
        # Simulate drift
        sartre.soma.drift_direction = -0.7f0
        sartre.soma.drift_speed = 0.6f0
        println("Drift simulated: past-oriented, high speed")

    elseif command == "/voice"
        if length(parts) > 1
            voice = Symbol(parts[2])
            obs = observe(sartre; voice=voice)
            println("\n$obs\n")
        else
            println("Available voices: observes, whispers, warns, wonders, mourns, hopes, confesses, celebrates, suspects, feels")
        end

    elseif command == "/help"
        println("""
        Commands:
          /state    - Show current soma state
          /feel     - Show current percept
          /trauma   - Simulate trauma state
          /peace    - Simulate peaceful state
          /void     - Simulate void state
          /drift    - Simulate emotional drift
          /voice X  - Use specific voice (observes, whispers, warns, etc.)
          /help     - Show this help
        """)

    else
        println("Unknown command: $command. Type /help for help.")
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# SERVER MODE (JSON-RPC)
# ═══════════════════════════════════════════════════════════════════════════════

function server_mode()
    println(JSON3.write(Dict("status" => "ready", "version" => "1.0")))
    flush(stdout)

    sartre = SARTRE()

    for line in eachline(stdin)
        line = strip(line)
        isempty(line) && continue

        try
            request = JSON3.read(line, Dict{String, Any})
            response = process_query(sartre, request)
            println(JSON3.write(response))
            flush(stdout)

            # Check for quit
            if get(request, "command", "") in ["quit", "exit"]
                break
            end
        catch e
            println(JSON3.write(Dict("status" => "error", "message" => string(e))))
            flush(stdout)
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST MODE
# ═══════════════════════════════════════════════════════════════════════════════

function test_mode()
    println("\n" * "="^70)
    println("  SARTRE TESTS")
    println("="^70 * "\n")

    passed = 0
    failed = 0

    # Test 1: AriannaSoma creation
    print("Test AriannaSoma creation... ")
    try
        soma = AriannaSoma()
        @assert soma.arousal == 0.3f0
        @assert soma.coherence == 0.8f0
        println("PASS")
        passed += 1
    catch e
        println("FAIL: $e")
        failed += 1
    end

    # Test 2: Feel state
    print("Test feel_state... ")
    try
        soma = AriannaSoma()
        percept = feel_state(soma)
        @assert 0 <= percept.warmth <= 1
        @assert 0 <= percept.clarity <= 1
        println("PASS")
        passed += 1
    catch e
        println("FAIL: $e")
        failed += 1
    end

    # Test 3: Voice selection
    print("Test voice selection... ")
    try
        soma = AriannaSoma()
        soma.trauma_level = 0.8f0
        voice = SARTREKernel.select_voice(soma)
        @assert voice == :warns
        println("PASS")
        passed += 1
    catch e
        println("FAIL: $e")
        failed += 1
    end

    # Test 4: Generate observation
    print("Test observation generation... ")
    try
        sartre = SARTRE()
        obs = observe(sartre)
        @assert length(obs) > 0
        @assert occursin("SARTRE", obs)
        println("PASS")
        passed += 1
    catch e
        println("FAIL: $e")
        failed += 1
    end

    # Test 5: Trauma state changes voice
    print("Test trauma voice switch... ")
    try
        sartre = SARTRE()
        sartre.soma.trauma_level = 0.7f0
        obs = observe(sartre)
        @assert occursin("warns", obs) || occursin("feels", obs)
        println("PASS")
        passed += 1
    catch e
        println("FAIL: $e")
        failed += 1
    end

    # Test 6: Peaceful state
    print("Test peaceful voice... ")
    try
        sartre = SARTRE()
        sartre.soma.coherence = 0.9f0
        sartre.soma.arousal = 0.1f0
        sartre.soma.trauma_level = 0.0f0
        obs = observe(sartre)
        @assert occursin("SARTRE", obs)
        println("PASS")
        passed += 1
    catch e
        println("FAIL: $e")
        failed += 1
    end

    # Test 7: JSON update
    print("Test JSON state update... ")
    try
        soma = AriannaSoma()
        json = """{"arousal": 0.9, "valence": -0.5, "coherence": 0.3}"""
        read_inner_world!(soma, json)
        @assert soma.arousal == 0.9f0
        @assert soma.valence == -0.5f0
        println("PASS")
        passed += 1
    catch e
        println("FAIL: $e")
        failed += 1
    end

    println("\n" * "="^70)
    println("  Results: $passed passed, $failed failed")
    println("="^70)

    return failed == 0
end

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

function main()
    if "--serve" in ARGS
        server_mode()
    elseif "--test" in ARGS
        test_mode() || exit(1)
    elseif "--train" in ARGS
        include("train.jl")
        main()  # Call train main
    else
        interactive_mode()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
