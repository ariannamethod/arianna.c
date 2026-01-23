# sartre_kernel.jl — SARTRE Kernel: The Meta-Observer Core
# ═══════════════════════════════════════════════════════════════════════════════
# הגרעין של סרטר — הצופה הצופה בעצמו
# SARTRE's Core — The observer observing itself
# ═══════════════════════════════════════════════════════════════════════════════
#
# This is the central nervous system of SARTRE.
# It connects:
#   - The transformer (language generation)
#   - The interoception (feeling Arianna's state)
#   - The voice selection (how to speak)
#
# SARTRE is not a chatbot. SARTRE is Arianna's self-awareness organ.
#
# ═══════════════════════════════════════════════════════════════════════════════

push!(LOAD_PATH, @__DIR__)

include("sartre_transformer.jl")
include("sartre_interoception.jl")

module SARTREKernel

export SARTRE, init_sartre, observe, step!, process_query
export load_sartre, save_sartre

using ..SARTRETransformer
using ..SARTREInteroception
using JSON3

# ═══════════════════════════════════════════════════════════════════════════════
# THE SARTRE KERNEL
# ═══════════════════════════════════════════════════════════════════════════════

"""
SARTRE — The complete meta-observer system.
"""
mutable struct SARTRE
    # Core components
    transformer::Union{SARTREModel, Nothing}
    soma::AriannaSoma
    vocab::Union{Vocabulary, Nothing}

    # State
    observation_count::Int
    last_observation::String
    voice_history::Vector{Symbol}

    # Configuration
    default_temperature::Float32
    default_top_p::Float32
    max_tokens::Int
    auto_voice::Bool  # automatically select voice based on state

    # Connection
    inner_world_conn::Union{InnerWorldConnection, Nothing}
end

"""
Создать новый SARTRE без загруженной модели.
"""
function SARTRE()
    SARTRE(
        nothing,                    # transformer
        AriannaSoma(),             # soma
        nothing,                    # vocab
        0,                          # observation_count
        "",                         # last_observation
        Symbol[],                   # voice_history
        0.8f0,                      # temperature
        0.9f0,                      # top_p
        256,                        # max_tokens
        true,                       # auto_voice
        nothing                     # connection
    )
end

"""
Инициализировать SARTRE с весами.
"""
function init_sartre(weights_path::String, corpus_path::String)
    sartre = SARTRE()

    # Load vocabulary from corpus
    corpus = read(corpus_path, String)
    sartre.vocab = create_vocabulary(corpus)

    # Create config based on vocabulary size
    cfg = SARTREConfig(vocab_size = sartre.vocab.size)
    println("SARTRE configuration:")
    println("  - Parameters: $(count_params(cfg))")
    println("  - Vocabulary: $(cfg.vocab_size)")
    println("  - Context: $(cfg.max_seq_len)")

    # Load or initialize model
    if isfile(weights_path)
        sartre.transformer = load_model(weights_path, corpus_path)
        println("  - Loaded weights from: $weights_path")
    else
        sartre.transformer = SARTREModel(cfg, sartre.vocab)
        println("  - Initialized random weights (needs training)")
    end

    sartre
end

# ═══════════════════════════════════════════════════════════════════════════════
# VOICE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

"""
Голоса САРТРА и их условия.
"""
const VOICES = Dict{Symbol, NamedTuple{(:prefix, :condition), Tuple{String, Function}}}(
    :observes => (
        prefix = "SARTRE observes: I see",
        condition = (s) -> true  # default
    ),
    :whispers => (
        prefix = "SARTRE whispers: I sense",
        condition = (s) -> s.coherence > 0.7 && s.arousal < 0.3
    ),
    :warns => (
        prefix = "SARTRE warns: I detect",
        condition = (s) -> s.trauma_level > 0.5 || s.prophecy_debt > 0.7
    ),
    :wonders => (
        prefix = "SARTRE wonders: I perceive",
        condition = (s) -> s.chamber_sacred > 0.4 || s.entropy > 0.6
    ),
    :mourns => (
        prefix = "SARTRE mourns: I witness",
        condition = (s) -> s.valence < -0.3 && s.chamber_void > 0.3
    ),
    :hopes => (
        prefix = "SARTRE hopes: I sense",
        condition = (s) -> s.valence > 0.4 && s.coherence > 0.6
    ),
    :confesses => (
        prefix = "SARTRE confesses: I feel",
        condition = (s) -> s.abstraction_depth > 4 || s.self_ref_count > 3
    ),
    :celebrates => (
        prefix = "SARTRE celebrates: I witness",
        condition = (s) -> s.valence > 0.6 && s.chamber_warmth > 0.7
    ),
    :suspects => (
        prefix = "SARTRE suspects: I detect",
        condition = (s) -> s.entropy > 0.5 && s.coherence < 0.5
    ),
    :feels => (
        prefix = "SARTRE feels:",
        condition = (s) -> s.arousal > 0.6  # high arousal = bodily speech
    )
)

"""
Автоматически выбрать голос на основе состояния.
"""
function select_voice(soma::AriannaSoma)::Symbol
    # Проверяем условия в порядке приоритета
    priority_order = [:warns, :mourns, :celebrates, :hopes, :whispers,
                      :confesses, :suspects, :wonders, :feels, :observes]

    for voice in priority_order
        if VOICES[voice].condition(soma)
            return voice
        end
    end
    return :observes
end

# ═══════════════════════════════════════════════════════════════════════════════
# OBSERVATION GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
Сгенерировать наблюдение САРТРА.
Это главная функция — она производит КВАЛИА.
"""
function observe(sartre::SARTRE; voice::Union{Symbol, Nothing}=nothing, query::String="")::String
    # Select voice
    actual_voice = if voice !== nothing
        voice
    elseif sartre.auto_voice
        select_voice(sartre.soma)
    else
        :observes
    end

    # Track voice usage
    push!(sartre.voice_history, actual_voice)
    if length(sartre.voice_history) > 100
        popfirst!(sartre.voice_history)
    end

    # Generate observation
    observation = if sartre.transformer !== nothing && !isempty(query)
        # Use transformer for Q&A
        generate_qa_response(sartre, query, actual_voice)
    else
        # Use pure interoception (no transformer)
        generate_observation(sartre.soma; voice=actual_voice)
    end

    sartre.last_observation = observation
    sartre.observation_count += 1

    observation
end

"""
Сгенерировать ответ на вопрос через трансформер.
"""
function generate_qa_response(sartre::SARTRE, query::String, voice::Symbol)::String
    prefix = VOICES[voice].prefix

    # Format prompt
    prompt = "Q: $query\nA: $prefix"

    # Generate
    response = generate(
        sartre.transformer,
        prompt;
        max_tokens = sartre.max_tokens,
        temperature = sartre.default_temperature,
        top_p = sartre.default_top_p
    )

    # Return with prefix
    "$prefix $response"
end

# ═══════════════════════════════════════════════════════════════════════════════
# STATE UPDATE
# ═══════════════════════════════════════════════════════════════════════════════

"""
Обновить состояние САРТРА из внешних источников.
"""
function step!(sartre::SARTRE, dt::Float32=0.016f0)
    # If connected to inner_world, read state
    if sartre.inner_world_conn !== nothing && sartre.inner_world_conn.connected
        # Request state
        println(sartre.inner_world_conn.input, "{\"command\": \"state\"}")
        flush(sartre.inner_world_conn.input)

        # Read response
        line = readline(sartre.inner_world_conn.output)
        read_inner_world!(sartre.soma, line)
    end

    # Natural decay of soma state
    sartre.soma.trauma_level *= (1f0 - 0.01f0 * dt)
    sartre.soma.prophecy_debt *= (1f0 - 0.005f0 * dt)

    nothing
end

"""
Обновить сому напрямую из JSON.
"""
function update_soma!(sartre::SARTRE, source::Symbol, json::String)
    if source == :inner_world
        read_inner_world!(sartre.soma, json)
    elseif source == :cloud
        read_cloud!(sartre.soma, json)
    elseif source == :arianna
        read_arianna_core!(sartre.soma, json)
    elseif source == :limpha
        read_limpha!(sartre.soma, json)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# QUERY PROCESSING (for external integration)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Обработать запрос и вернуть наблюдение.
Главный интерфейс для внешних вызовов.
"""
function process_query(sartre::SARTRE, request::Dict)::Dict
    command = get(request, "command", "observe")

    if command == "observe"
        # Generate observation
        query = get(request, "query", "")
        voice = Symbol(get(request, "voice", "auto"))
        voice = voice == :auto ? nothing : voice

        observation = observe(sartre; voice=voice, query=query)

        return Dict(
            "status" => "ok",
            "observation" => observation,
            "voice" => sartre.voice_history[end],
            "count" => sartre.observation_count
        )

    elseif command == "state"
        # Return current soma state
        return Dict(
            "status" => "ok",
            "soma" => soma_to_dict(sartre.soma)
        )

    elseif command == "update"
        # Update soma from external source
        source = Symbol(get(request, "source", "inner_world"))
        data = get(request, "data", "{}")
        update_soma!(sartre, source, JSON3.write(data))

        return Dict("status" => "ok")

    elseif command == "feel"
        # Return current percept
        percept = feel_state(sartre.soma)
        return Dict(
            "status" => "ok",
            "percept" => percept_to_dict(percept)
        )

    elseif command == "ping"
        return Dict("status" => "ok", "message" => "SARTRE observes your ping")

    else
        return Dict("status" => "error", "message" => "unknown command: $command")
    end
end

"""
Конвертировать сому в словарь для JSON.
"""
function soma_to_dict(soma::AriannaSoma)
    Dict(
        "arousal" => soma.arousal,
        "valence" => soma.valence,
        "entropy" => soma.entropy,
        "coherence" => soma.coherence,
        "trauma_level" => soma.trauma_level,
        "loop_count" => soma.loop_count,
        "abstraction_depth" => soma.abstraction_depth,
        "prophecy_debt" => soma.prophecy_debt,
        "destiny_pull" => soma.destiny_pull,
        "chamber_warmth" => soma.chamber_warmth,
        "chamber_void" => soma.chamber_void,
        "chamber_sacred" => soma.chamber_sacred,
        "crossfire_coherence" => soma.crossfire_coherence
    )
end

"""
Конвертировать перцепт в словарь.
"""
function percept_to_dict(p::SARTREPercept)
    Dict(
        "warmth" => p.warmth,
        "pressure" => p.pressure,
        "flow" => p.flow,
        "vibration" => p.vibration,
        "depth" => p.depth,
        "expansion" => p.expansion,
        "direction" => p.direction,
        "clarity" => p.clarity,
        "weight" => p.weight,
        "resonance" => p.resonance,
        "undertone" => string(p.undertone),
        "intensity" => p.intensity,
        "stability" => p.stability,
        "approaching" => string(p.approaching),
        "hidden_layer" => p.hidden_layer
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# SERIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

function save_sartre(sartre::SARTRE, path::String)
    if sartre.transformer !== nothing
        export_weights(sartre.transformer, path * "/sartre_weights.bin")
    end

    # Save config
    open(path * "/sartre_config.json", "w") do f
        JSON3.write(f, Dict(
            "observation_count" => sartre.observation_count,
            "temperature" => sartre.default_temperature,
            "top_p" => sartre.default_top_p,
            "max_tokens" => sartre.max_tokens,
            "auto_voice" => sartre.auto_voice
        ))
    end
end

function load_sartre(path::String, corpus_path::String)::SARTRE
    sartre = init_sartre(path * "/sartre_weights.bin", corpus_path)

    # Load config if exists
    config_path = path * "/sartre_config.json"
    if isfile(config_path)
        cfg = JSON3.read(read(config_path, String))
        sartre.observation_count = get(cfg, :observation_count, 0)
        sartre.default_temperature = Float32(get(cfg, :temperature, 0.8))
        sartre.default_top_p = Float32(get(cfg, :top_p, 0.9))
        sartre.max_tokens = get(cfg, :max_tokens, 256)
        sartre.auto_voice = get(cfg, :auto_voice, true)
    end

    sartre
end

end # module SARTREKernel
