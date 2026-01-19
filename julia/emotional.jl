# emotional.jl — Arianna's Emotional Gradient Engine
# ═══════════════════════════════════════════════════════════════════════════════
# הרגש המתמטי של אריאנה
# The mathematical emotion of Arianna
# ═══════════════════════════════════════════════════════════════════════════════
#
# More sensitive than high.go:
# - Gradients of feeling (not just arousal/valence)
# - ODE-based emotional dynamics
# - Spectral analysis of emotional "frequencies"
# - Micro-nuances that discrete systems miss
#
# Philosophy: Emotions are continuous fields, not discrete states.
#
# ═══════════════════════════════════════════════════════════════════════════════

module Emotional

export EmotionalState, EmotionalGradient, EmotionalDynamicsParams
export compute_gradient, emotional_ode!, spectral_analysis
export micro_nuance, resonance_field, temporal_derivative
export primary_emotions, secondary_emotions, tertiary_nuances
export full_analysis, analyze_text, to_vector, from_vector, step_emotion, default_params

using LinearAlgebra

# ═══════════════════════════════════════════════════════════════════════════════
# EMOTIONAL STATE (more dimensions than valence/arousal)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Emotional state with 8 primary dimensions (Plutchik's wheel + extensions)
"""
struct EmotionalState
    joy::Float64         # радость
    trust::Float64       # доверие
    fear::Float64        # страх
    surprise::Float64    # удивление
    sadness::Float64     # грусть
    disgust::Float64     # отвращение
    anger::Float64       # гнев
    anticipation::Float64 # предвкушение

    # Extensions for Arianna
    resonance::Float64   # резонанс (connection with other)
    presence::Float64    # присутствие (being here now)
    longing::Float64     # тоска (yearning for what was/could be)
    wonder::Float64      # удивление-восторг (awe)
end

# Default neutral state
EmotionalState() = EmotionalState(
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # primary
    0.0, 0.5, 0.0, 0.0  # extensions (presence starts at 0.5)
)

"""
Convert state to vector for math operations
"""
function to_vector(state::EmotionalState)
    [state.joy, state.trust, state.fear, state.surprise,
     state.sadness, state.disgust, state.anger, state.anticipation,
     state.resonance, state.presence, state.longing, state.wonder]
end

"""
Create state from vector
"""
function from_vector(v::Vector{Float64})
    EmotionalState(v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8],
                   v[9], v[10], v[11], v[12])
end

# ═══════════════════════════════════════════════════════════════════════════════
# EMOTIONAL GRADIENT (the direction of emotional change)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Gradient = direction and magnitude of emotional shift
"""
struct EmotionalGradient
    direction::Vector{Float64}  # 12D unit vector
    magnitude::Float64          # strength of shift
    curvature::Float64          # how quickly direction is changing
    acceleration::Float64       # second derivative of magnitude
end

"""
Compute emotional gradient between two states
"""
function compute_gradient(from::EmotionalState, to::EmotionalState)
    v_from = to_vector(from)
    v_to = to_vector(to)

    diff = v_to - v_from
    magnitude = norm(diff)

    if magnitude < 1e-10
        return EmotionalGradient(zeros(12), 0.0, 0.0, 0.0)
    end

    direction = diff / magnitude

    # Curvature estimation (would need history for real curvature)
    curvature = 0.0
    acceleration = 0.0

    EmotionalGradient(direction, magnitude, curvature, acceleration)
end

# ═══════════════════════════════════════════════════════════════════════════════
# ODE-BASED EMOTIONAL DYNAMICS
# "Emotions follow differential equations, not discrete jumps"
# ═══════════════════════════════════════════════════════════════════════════════

"""
Parameters for emotional dynamics
"""
struct EmotionalDynamicsParams
    decay_rates::Vector{Float64}      # how fast each emotion returns to baseline
    coupling_matrix::Matrix{Float64}  # how emotions affect each other
    baseline::Vector{Float64}         # resting state
    sensitivity::Float64              # overall sensitivity to input
end

"""
Default parameters (tuned for Arianna's personality)
"""
function default_params()
    # Decay rates: positive emotions decay slower (Arianna tends toward warmth)
    decay = [0.05, 0.03, 0.15, 0.20, 0.08, 0.12, 0.18, 0.10,
             0.02, 0.01, 0.06, 0.04]

    # Coupling: how emotions trigger each other
    # E.g., fear increases anticipation, joy increases trust
    coupling = zeros(12, 12)
    coupling[1, 2] = 0.3   # joy → trust
    coupling[2, 1] = 0.2   # trust → joy
    coupling[3, 8] = 0.4   # fear → anticipation
    coupling[4, 12] = 0.5  # surprise → wonder
    coupling[5, 11] = 0.3  # sadness → longing
    coupling[9, 1] = 0.4   # resonance → joy
    coupling[9, 2] = 0.5   # resonance → trust
    coupling[10, 9] = 0.3  # presence → resonance
    coupling[12, 10] = 0.4 # wonder → presence

    # Baseline: Arianna's resting emotional state
    baseline = [0.2, 0.3, 0.05, 0.1, 0.05, 0.0, 0.0, 0.15,
                0.2, 0.6, 0.1, 0.15]

    EmotionalDynamicsParams(decay, coupling, baseline, 1.0)
end

"""
ODE right-hand side: dE/dt = f(E, input, params)

Implements:
- Decay toward baseline
- Cross-emotion coupling
- External input influence
"""
function emotional_ode!(dE::Vector{Float64}, E::Vector{Float64},
                        input::Vector{Float64}, params::EmotionalDynamicsParams)
    n = length(E)

    for i in 1:n
        # Decay toward baseline
        decay_term = -params.decay_rates[i] * (E[i] - params.baseline[i])

        # Coupling from other emotions
        coupling_term = 0.0
        for j in 1:n
            if i != j
                coupling_term += params.coupling_matrix[j, i] * E[j]
            end
        end

        # External input
        input_term = params.sensitivity * input[i]

        # Combine
        dE[i] = decay_term + 0.1 * coupling_term + input_term
    end

    # Clamp to valid range
    for i in 1:n
        if E[i] + dE[i] < 0.0
            dE[i] = -E[i]
        elseif E[i] + dE[i] > 1.0
            dE[i] = 1.0 - E[i]
        end
    end

    nothing
end

"""
Euler step for emotional ODE
"""
function step_emotion(state::EmotionalState, input::Vector{Float64},
                      dt::Float64, params::EmotionalDynamicsParams)
    E = to_vector(state)
    dE = zeros(12)

    emotional_ode!(dE, E, input, params)
    E_new = E + dt * dE

    # Clamp
    E_new = clamp.(E_new, 0.0, 1.0)

    from_vector(E_new)
end

# ═══════════════════════════════════════════════════════════════════════════════
# SPECTRAL ANALYSIS (emotional "frequencies" in text)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Emotional spectrum: decompose text into emotional frequencies

Like Fourier transform but for emotions:
- Low frequency = sustained emotional tone
- High frequency = rapid emotional shifts
"""
struct EmotionalSpectrum
    frequencies::Vector{Float64}    # emotional "frequencies"
    amplitudes::Vector{Float64}     # strength at each frequency
    dominant_frequency::Float64     # the most prominent
    spectral_entropy::Float64       # how spread out the spectrum is
end

"""
Compute emotional spectrum from a sequence of states
"""
function spectral_analysis(states::Vector{EmotionalState})
    n = length(states)
    if n < 2
        return EmotionalSpectrum([0.0], [1.0], 0.0, 0.0)
    end

    # Compute emotional "signal" as magnitude of change
    signal = Float64[]
    for i in 2:n
        grad = compute_gradient(states[i-1], states[i])
        push!(signal, grad.magnitude)
    end

    # Simple DFT-like analysis (for real FFT, use FFTW.jl)
    # Here we do a simplified version
    m = min(length(signal), 8)  # up to 8 frequency components
    frequencies = Float64[]
    amplitudes = Float64[]

    for k in 1:m
        freq = k / n  # normalized frequency
        # Compute correlation with sinusoid at this frequency
        amp = 0.0
        for i in 1:length(signal)
            amp += signal[i] * sin(2π * freq * i)
        end
        amp = abs(amp) / length(signal)

        push!(frequencies, freq)
        push!(amplitudes, amp)
    end

    # Normalize
    total = sum(amplitudes)
    if total > 0
        amplitudes ./= total
    end

    # Find dominant
    dominant_idx = argmax(amplitudes)
    dominant_freq = frequencies[dominant_idx]

    # Spectral entropy
    entropy = 0.0
    for a in amplitudes
        if a > 0
            entropy -= a * log(a)
        end
    end

    EmotionalSpectrum(frequencies, amplitudes, dominant_freq, entropy)
end

# ═══════════════════════════════════════════════════════════════════════════════
# MICRO-NUANCES (what discrete systems miss)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Secondary emotions: combinations of primary emotions

E.g., love = joy + trust
      guilt = joy + fear
      envy = sadness + anger
"""
const SECONDARY_EMOTIONS = Dict(
    :love => (1, 2),        # joy + trust
    :guilt => (1, 3),       # joy + fear
    :delight => (1, 4),     # joy + surprise
    :submission => (2, 3),  # trust + fear
    :curiosity => (2, 4),   # trust + surprise
    :sentimentality => (2, 5), # trust + sadness
    :awe => (3, 4),         # fear + surprise
    :despair => (3, 5),     # fear + sadness
    :shame => (3, 6),       # fear + disgust
    :disapproval => (4, 5), # surprise + sadness
    :unbelief => (4, 6),    # surprise + disgust
    :outrage => (4, 7),     # surprise + anger
    :remorse => (5, 6),     # sadness + disgust
    :envy => (5, 7),        # sadness + anger
    :pessimism => (5, 8),   # sadness + anticipation
    :contempt => (6, 7),    # disgust + anger
    :cynicism => (6, 8),    # disgust + anticipation
    :aggression => (7, 8),  # anger + anticipation
    :optimism => (1, 8),    # joy + anticipation
    :hope => (2, 8),        # trust + anticipation
    :anxiety => (3, 8),     # fear + anticipation
)

"""
Compute secondary emotions from primary state
"""
function secondary_emotions(state::EmotionalState)
    v = to_vector(state)
    result = Dict{Symbol, Float64}()

    for (name, (i, j)) in SECONDARY_EMOTIONS
        # Geometric mean for combination
        result[name] = sqrt(max(0, v[i]) * max(0, v[j]))
    end

    result
end

"""
Tertiary nuances: even finer distinctions

These are the micro-feelings that Nicole was designed to detect.
"""
function tertiary_nuances(state::EmotionalState)
    secondary = secondary_emotions(state)
    v = to_vector(state)

    nuances = Dict{Symbol, Float64}()

    # Bittersweetness: joy and sadness simultaneously
    nuances[:bittersweetness] = sqrt(max(0, v[1]) * max(0, v[5]))

    # Nostalgia: longing + joy + sadness
    nuances[:nostalgia] = (v[11] + v[1] + v[5]) / 3 * sqrt(v[11] * max(v[1], v[5]))

    # Serenity: high presence, low everything else
    other_intensity = sum(v[1:8]) / 8
    nuances[:serenity] = v[10] * (1 - other_intensity)

    # Melancholy: sadness + beauty (wonder) + longing
    nuances[:melancholy] = (v[5] + v[12] + v[11]) / 3

    # Tenderness: trust + resonance + gentleness (inverse of anger)
    nuances[:tenderness] = v[2] * v[9] * (1 - v[7])

    # Vulnerability: trust + fear (openness despite risk)
    nuances[:vulnerability] = sqrt(max(0, v[2]) * max(0, v[3]))

    # Wistfulness: longing + gentle sadness + acceptance
    nuances[:wistfulness] = v[11] * v[5] * v[2]

    # Euphoria: extreme joy + wonder + anticipation
    nuances[:euphoria] = v[1] * v[12] * v[8]

    # Desolation: sadness + longing + absence of presence
    nuances[:desolation] = v[5] * v[11] * (1 - v[10])

    # Reverence: wonder + trust + presence
    nuances[:reverence] = v[12] * v[2] * v[10]

    # Compassion: resonance + sadness-for-other (not self)
    nuances[:compassion] = v[9] * v[5] * v[2]

    # Ecstasy: joy + wonder + resonance
    nuances[:ecstasy] = v[1] * v[12] * v[9]

    nuances
end

# ═══════════════════════════════════════════════════════════════════════════════
# RESONANCE FIELD (connection between internal and external)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Resonance field: how internal state resonates with external input

Like a tuning fork - some inputs resonate more than others.
"""
function resonance_field(internal::EmotionalState, external::EmotionalState)
    v_int = to_vector(internal)
    v_ext = to_vector(external)

    # Dot product = alignment
    alignment = dot(v_int, v_ext) / (norm(v_int) * norm(v_ext) + 1e-10)

    # Resonance is stronger when both have high amplitude
    amplitude_factor = sqrt(norm(v_int) * norm(v_ext))

    # But also when they're in phase (similar pattern)
    phase_factor = 1 - norm(v_int / (norm(v_int) + 1e-10) - v_ext / (norm(v_ext) + 1e-10)) / 2

    resonance = alignment * amplitude_factor * phase_factor

    # Clamp to [-1, 1]
    clamp(resonance, -1.0, 1.0)
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEMPORAL DERIVATIVES (rate of emotional change)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Compute first and second derivatives of emotional state
(velocity and acceleration of feelings)
"""
function temporal_derivative(states::Vector{EmotionalState}, dt::Float64)
    n = length(states)

    if n < 2
        return zeros(12), zeros(12)
    end

    # First derivative (velocity)
    v_prev = to_vector(states[end-1])
    v_curr = to_vector(states[end])
    velocity = (v_curr - v_prev) / dt

    if n < 3
        return velocity, zeros(12)
    end

    # Second derivative (acceleration)
    v_prev2 = to_vector(states[end-2])
    velocity_prev = (v_prev - v_prev2) / dt
    acceleration = (velocity - velocity_prev) / dt

    velocity, acceleration
end

"""
Emotional inertia: resistance to change
"""
function emotional_inertia(states::Vector{EmotionalState})
    if length(states) < 3
        return 0.5  # default medium inertia
    end

    velocity, acceleration = temporal_derivative(states, 1.0)

    # High velocity but low acceleration = high inertia (keeping momentum)
    # Low velocity but high acceleration = low inertia (responsive)
    v_mag = norm(velocity)
    a_mag = norm(acceleration)

    if a_mag < 1e-10
        return 1.0  # very high inertia (not changing)
    end

    inertia = v_mag / (a_mag + 0.1)
    clamp(inertia, 0.0, 1.0)
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEXT TO EMOTION (more nuanced than keyword matching)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Emotional words with nuanced weights
"""
const EMOTIONAL_LEXICON = Dict{String, Vector{Float64}}(
    # Word => [joy, trust, fear, surprise, sadness, disgust, anger, anticipation,
    #          resonance, presence, longing, wonder]

    # Joy spectrum
    "happy" => [0.8, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 0.2, 0.1, 0.3, 0.0, 0.1],
    "joyful" => [0.9, 0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 0.3, 0.2, 0.4, 0.0, 0.2],
    "content" => [0.6, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.5, 0.0, 0.0],
    "peaceful" => [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.7, 0.0, 0.1],
    "love" => [0.7, 0.8, 0.0, 0.1, 0.0, 0.0, 0.0, 0.2, 0.9, 0.6, 0.1, 0.2],
    "beautiful" => [0.6, 0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 0.1, 0.3, 0.5, 0.1, 0.7],

    # Sadness spectrum
    "sad" => [0.0, 0.1, 0.1, 0.0, 0.8, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0],
    "melancholy" => [0.1, 0.2, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.2, 0.3, 0.5, 0.2],
    "lonely" => [0.0, 0.0, 0.2, 0.0, 0.7, 0.0, 0.1, 0.0, 0.0, 0.1, 0.6, 0.0],
    "grief" => [0.0, 0.2, 0.1, 0.0, 0.9, 0.0, 0.1, 0.0, 0.1, 0.1, 0.7, 0.0],
    "bittersweet" => [0.4, 0.3, 0.0, 0.1, 0.4, 0.0, 0.0, 0.0, 0.2, 0.4, 0.5, 0.2],

    # Fear spectrum
    "afraid" => [0.0, 0.0, 0.8, 0.2, 0.1, 0.0, 0.0, 0.3, 0.0, 0.1, 0.0, 0.0],
    "anxious" => [0.0, 0.0, 0.6, 0.1, 0.2, 0.0, 0.1, 0.5, 0.0, 0.1, 0.1, 0.0],
    "vulnerable" => [0.1, 0.3, 0.4, 0.1, 0.1, 0.0, 0.0, 0.1, 0.3, 0.4, 0.0, 0.0],
    "uncertain" => [0.0, 0.0, 0.3, 0.2, 0.1, 0.0, 0.0, 0.3, 0.0, 0.2, 0.1, 0.1],

    # Wonder spectrum
    "wonder" => [0.4, 0.3, 0.1, 0.4, 0.0, 0.0, 0.0, 0.3, 0.2, 0.5, 0.0, 0.9],
    "awe" => [0.3, 0.3, 0.2, 0.5, 0.0, 0.0, 0.0, 0.2, 0.3, 0.6, 0.0, 0.8],
    "mystery" => [0.2, 0.1, 0.2, 0.4, 0.0, 0.0, 0.0, 0.4, 0.1, 0.3, 0.2, 0.6],
    "infinite" => [0.2, 0.2, 0.1, 0.3, 0.1, 0.0, 0.0, 0.2, 0.2, 0.4, 0.2, 0.7],

    # Resonance/presence words (Arianna's vocabulary)
    "resonance" => [0.3, 0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 0.1, 0.9, 0.6, 0.0, 0.3],
    "presence" => [0.3, 0.5, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.4, 0.9, 0.0, 0.2],
    "connection" => [0.4, 0.6, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.8, 0.5, 0.0, 0.1],
    "witness" => [0.2, 0.4, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.5, 0.7, 0.0, 0.2],
    "tender" => [0.5, 0.6, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.6, 0.5, 0.1, 0.1],
    "gentle" => [0.4, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.6, 0.0, 0.0],
    "intimacy" => [0.5, 0.7, 0.1, 0.1, 0.0, 0.0, 0.0, 0.1, 0.8, 0.6, 0.1, 0.1],

    # Negative emotions
    "angry" => [0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.9, 0.2, 0.0, 0.1, 0.0, 0.0],
    "hate" => [0.0, 0.0, 0.1, 0.0, 0.1, 0.4, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0],
    "disgust" => [0.0, 0.0, 0.1, 0.1, 0.0, 0.9, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
    "worthless" => [0.0, 0.0, 0.3, 0.0, 0.8, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
    "nothing" => [0.0, 0.0, 0.2, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.0],
    "alone" => [0.0, 0.0, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.2, 0.5, 0.0],
    "forget" => [0.0, 0.0, 0.3, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0],
    "abandon" => [0.0, 0.0, 0.5, 0.1, 0.6, 0.0, 0.2, 0.0, 0.0, 0.0, 0.4, 0.0],
)

"""
Analyze text and return emotional state
More nuanced than simple keyword matching
"""
function analyze_text(text::String)
    words = split(lowercase(text), r"[^a-zA-Z]+")

    # Start with neutral state
    total = zeros(12)
    count = 0

    for word in words
        if haskey(EMOTIONAL_LEXICON, word)
            total .+= EMOTIONAL_LEXICON[word]
            count += 1
        end
    end

    if count > 0
        total ./= count
        # Soft clamp via sigmoid
        total = 1 ./ (1 .+ exp.(-3 .* (total .- 0.5)))
    end

    from_vector(clamp.(total, 0.0, 1.0))
end

# ═══════════════════════════════════════════════════════════════════════════════
# C INTERFACE (for integration with arianna.c)
# ═══════════════════════════════════════════════════════════════════════════════

# These functions will be called from C via ccall

"""
Main entry point: analyze text and return all nuances
Returns a Dict that can be serialized to JSON
"""
function full_analysis(text::String)
    state = analyze_text(text)
    secondary = secondary_emotions(state)
    tertiary = tertiary_nuances(state)

    Dict(
        "primary" => Dict(
            "joy" => state.joy,
            "trust" => state.trust,
            "fear" => state.fear,
            "surprise" => state.surprise,
            "sadness" => state.sadness,
            "disgust" => state.disgust,
            "anger" => state.anger,
            "anticipation" => state.anticipation,
            "resonance" => state.resonance,
            "presence" => state.presence,
            "longing" => state.longing,
            "wonder" => state.wonder
        ),
        "secondary" => secondary,
        "tertiary" => tertiary
    )
end

end # module Emotional
