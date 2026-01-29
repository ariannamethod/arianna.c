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
include("temporal.jl")
using .Emotional
using .Temporal
using JSON3

# Global temporal state (persistent across commands)
global_temporal_state = create_state()
global_temporal_params = default_params()

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
# TEMPORAL HANDLERS (from PITOMADOM)
# ═══════════════════════════════════════════════════════════════════════════════

function handle_temporal_step(data)
    global global_temporal_state, global_temporal_params

    manifested = Float64(get(data, :manifested, 0.0))
    destined = Float64(get(data, :destined, 0.0))
    dt = Float64(get(data, :dt, 0.1))

    # Update calendar dissonance if date provided
    if haskey(data, :year)
        year = Int(data[:year])
        month = Int(get(data, :month, 1))
        day = Int(get(data, :day, 1))
        global_temporal_state.calendar_dissonance = birthday_dissonance(year, month, day)
        global_temporal_state.calendar_phase = calendar_drift(year, month, day)
    end

    # Step the ODE
    step_temporal(global_temporal_state, global_temporal_params, manifested, destined, dt)

    Dict(
        "status" => "ok",
        "prophecy_debt" => global_temporal_state.prophecy_debt,
        "tension" => global_temporal_state.tension,
        "pain" => global_temporal_state.pain,
        "drift_direction" => global_temporal_state.drift_direction,
        "temporal_alpha" => global_temporal_state.temporal_alpha,
        "wormhole_probability" => global_temporal_state.wormhole_probability,
        "calendar_dissonance" => global_temporal_state.calendar_dissonance
    )
end

function handle_temporal_dissonance(data)
    global global_temporal_state

    year = Int(get(data, :year, 2026))
    month = Int(get(data, :month, 1))
    day = Int(get(data, :day, 29))

    dissonance = compute_dissonance(global_temporal_state, year, month, day)
    cal_drift = calendar_drift(year, month, day)
    bd_dissonance = birthday_dissonance(year, month, day)

    Dict(
        "status" => "ok",
        "total_dissonance" => dissonance,
        "calendar_drift" => cal_drift,
        "birthday_dissonance" => bd_dissonance
    )
end

function handle_temporal_mode(data)
    global global_temporal_state

    mode_str = get(data, :mode, "prophecy")
    mode = if mode_str == "prophecy"
        PROPHECY
    elseif mode_str == "retrodiction"
        RETRODICTION
    else
        SYMMETRIC
    end

    global_temporal_state.mode = mode

    bias = if mode == PROPHECY
        prophecy_bias(global_temporal_state)
    elseif mode == RETRODICTION
        retrodiction_bias(global_temporal_state)
    else
        symmetric_bias(global_temporal_state)
    end

    Dict(
        "status" => "ok",
        "mode" => string(mode),
        "bias" => bias,
        "temporal_alpha" => global_temporal_state.temporal_alpha
    )
end

function handle_temporal_reset(data)
    global global_temporal_state
    global_temporal_state = create_state()

    Dict(
        "status" => "ok",
        "message" => "temporal state reset"
    )
end

function handle_wormhole_check(data)
    global global_temporal_state, global_temporal_params

    prob = wormhole_probability(global_temporal_state, global_temporal_params)

    # Check if wormhole opens
    opened = rand() < prob

    Dict(
        "status" => "ok",
        "probability" => prob,
        "opened" => opened,
        "prophecy_debt" => global_temporal_state.prophecy_debt,
        "calendar_dissonance" => global_temporal_state.calendar_dissonance
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
        # Temporal commands (from PITOMADOM)
        elseif command == "temporal_step"
            handle_temporal_step(data)
        elseif command == "temporal_dissonance"
            handle_temporal_dissonance(data)
        elseif command == "temporal_mode"
            handle_temporal_mode(data)
        elseif command == "temporal_reset"
            handle_temporal_reset(data)
        elseif command == "wormhole_check"
            handle_wormhole_check(data)
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
