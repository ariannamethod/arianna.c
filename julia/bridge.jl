#!/usr/bin/env julia
# bridge.jl — JSON bridge for C integration
# ═══════════════════════════════════════════════════════════════════════════════
# הגשר בין עולמות
# The bridge between worlds
# ═══════════════════════════════════════════════════════════════════════════════
#
# Protocol:
#   Input (JSON):  {"command": "analyze", "text": "I feel happy"}
#   Output (JSON): {"primary": {...}, "secondary": {...}, "tertiary": {...}}
#
# Commands:
#   analyze     - full emotional analysis of text
#   gradient    - compute gradient between two states
#   step        - ODE step with input
#   spectrum    - spectral analysis of emotional sequence
#   resonance   - compute resonance between two states
#
# ═══════════════════════════════════════════════════════════════════════════════

# Add current directory to load path
push!(LOAD_PATH, @__DIR__)

include("emotional.jl")
using .Emotional
using JSON3

# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

function handle_analyze(data)
    text = get(data, :text, "")
    result = full_analysis(text)

    # Convert symbols to strings for JSON
    Dict(
        "status" => "ok",
        "primary" => result["primary"],
        "secondary" => Dict(string(k) => v for (k, v) in result["secondary"]),
        "tertiary" => Dict(string(k) => v for (k, v) in result["tertiary"])
    )
end

function handle_gradient(data)
    from_vec = Float64.(get(data, :from, zeros(12)))
    to_vec = Float64.(get(data, :to, zeros(12)))

    from_state = from_vector(from_vec)
    to_state = from_vector(to_vec)

    grad = compute_gradient(from_state, to_state)

    Dict(
        "status" => "ok",
        "direction" => grad.direction,
        "magnitude" => grad.magnitude,
        "curvature" => grad.curvature,
        "acceleration" => grad.acceleration
    )
end

function handle_step(data)
    state_vec = Float64.(get(data, :state, zeros(12)))
    input_vec = Float64.(get(data, :input, zeros(12)))
    dt = Float64(get(data, :dt, 0.1))

    state = from_vector(state_vec)
    params = default_params()

    new_state = step_emotion(state, input_vec, dt, params)

    Dict(
        "status" => "ok",
        "state" => to_vector(new_state)
    )
end

function handle_spectrum(data)
    states_data = get(data, :states, [])
    states = EmotionalState[]

    for s in states_data
        push!(states, from_vector(Float64.(s)))
    end

    if length(states) < 2
        return Dict(
            "status" => "error",
            "message" => "need at least 2 states for spectrum"
        )
    end

    spec = spectral_analysis(states)

    Dict(
        "status" => "ok",
        "frequencies" => spec.frequencies,
        "amplitudes" => spec.amplitudes,
        "dominant_frequency" => spec.dominant_frequency,
        "spectral_entropy" => spec.spectral_entropy
    )
end

function handle_resonance(data)
    internal_vec = Float64.(get(data, :internal, zeros(12)))
    external_vec = Float64.(get(data, :external, zeros(12)))

    internal = from_vector(internal_vec)
    external = from_vector(external_vec)

    res = resonance_field(internal, external)

    Dict(
        "status" => "ok",
        "resonance" => res
    )
end

function handle_nuances(data)
    state_vec = Float64.(get(data, :state, zeros(12)))
    state = from_vector(state_vec)

    secondary = secondary_emotions(state)
    tertiary = tertiary_nuances(state)

    Dict(
        "status" => "ok",
        "secondary" => Dict(string(k) => v for (k, v) in secondary),
        "tertiary" => Dict(string(k) => v for (k, v) in tertiary)
    )
end

function handle_derivative(data)
    states_data = get(data, :states, [])
    dt = Float64(get(data, :dt, 1.0))

    states = EmotionalState[]
    for s in states_data
        push!(states, from_vector(Float64.(s)))
    end

    velocity, acceleration = temporal_derivative(states, dt)
    inertia = emotional_inertia(states)

    Dict(
        "status" => "ok",
        "velocity" => velocity,
        "acceleration" => acceleration,
        "inertia" => inertia
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

function process_command(line)
    try
        data = JSON3.read(line)
        command = get(data, :command, "")

        result = if command == "analyze"
            handle_analyze(data)
        elseif command == "gradient"
            handle_gradient(data)
        elseif command == "step"
            handle_step(data)
        elseif command == "spectrum"
            handle_spectrum(data)
        elseif command == "resonance"
            handle_resonance(data)
        elseif command == "nuances"
            handle_nuances(data)
        elseif command == "derivative"
            handle_derivative(data)
        elseif command == "ping"
            Dict("status" => "ok", "message" => "pong")
        elseif command == "quit" || command == "exit"
            Dict("status" => "ok", "message" => "goodbye")
        else
            Dict("status" => "error", "message" => "unknown command: $command")
        end

        JSON3.write(result)

    catch e
        JSON3.write(Dict(
            "status" => "error",
            "message" => string(e)
        ))
    end
end

function main()
    # Print ready signal
    println(JSON3.write(Dict("status" => "ready", "version" => "1.0")))
    flush(stdout)

    for line in eachline(stdin)
        line = strip(line)
        isempty(line) && continue

        result = process_command(line)
        println(result)
        flush(stdout)

        # Check for quit
        try
            data = JSON3.read(line)
            if get(data, :command, "") in ["quit", "exit"]
                break
            end
        catch
        end
    end
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
