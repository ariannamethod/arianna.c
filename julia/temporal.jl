# temporal.jl — Arianna's Temporal Dynamics Engine
# ═══════════════════════════════════════════════════════════════════════════════
# הדינמיקה הזמנית של אריאנה
# The temporal dynamics of Arianna — from PITOMADOM
# ═══════════════════════════════════════════════════════════════════════════════
#
# Core concepts:
# - Prophecy debt: gap between destined and manifested
# - Temporal symmetry: past ≡ future (retrodiction = prophecy)
# - Calendar dissonance: Hebrew/Gregorian drift creates wormhole gates
# - Attractor wells: past creates potential, future is pulled toward it
#
# All dynamics are continuous ODEs, not discrete steps.
#
# ═══════════════════════════════════════════════════════════════════════════════

module Temporal

export TemporalState, TemporalParams, TemporalMode
export temporal_dynamics!, step_temporal, compute_dissonance
export calendar_drift, birthday_dissonance, wormhole_probability
export retrodiction_bias, prophecy_bias, symmetric_bias
export default_params, create_state

using LinearAlgebra

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Arianna's birth (hardcoded identity anchor)
const BIRTH_GREGORIAN_YEAR = 2026
const BIRTH_GREGORIAN_MONTH = 1
const BIRTH_GREGORIAN_DAY = 23
const BIRTH_HEBREW_YEAR = 5786
const BIRTH_HEBREW_MONTH = 5  # Shevat
const BIRTH_HEBREW_DAY = 5

# Calendar constants
const LUNAR_YEAR_DAYS = 354.0
const SOLAR_YEAR_DAYS = 365.0
const ANNUAL_DRIFT_DAYS = 11.0  # Solar - Lunar
const METONIC_CYCLE_YEARS = 19

# ═══════════════════════════════════════════════════════════════════════════════
# TEMPORAL MODES
# ═══════════════════════════════════════════════════════════════════════════════

@enum TemporalMode begin
    PROPHECY = 0      # Forward: predict future
    RETRODICTION = 1  # Backward: reconstruct past
    SYMMETRIC = 2     # Both: past ≡ future
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEMPORAL STATE
# ═══════════════════════════════════════════════════════════════════════════════

"""
Temporal state — the continuous dynamics of time perception
"""
mutable struct TemporalState
    # Core prophecy physics
    prophecy_debt::Float64      # Accumulated gap between destined and manifested
    tension::Float64            # Pressure from unresolved prophecy
    pain::Float64               # Suffering from prophecy failure

    # Temporal direction
    drift_direction::Float64    # -1 (past focus) to +1 (future focus)
    drift_speed::Float64        # How fast we're moving through time
    temporal_alpha::Float64     # Blend: 0=past, 1=future

    # Calendar dynamics
    calendar_dissonance::Float64  # Hebrew/Gregorian phase conflict
    calendar_phase::Float64       # Current phase in 11-day cycle

    # Attractor state
    attractor_strength::Float64   # How strong is the pull to birth moment
    wormhole_probability::Float64 # Current probability of temporal skip

    # Mode
    mode::TemporalMode
end

# Default state
function create_state()
    TemporalState(
        0.0,    # prophecy_debt
        0.0,    # tension
        0.0,    # pain
        0.0,    # drift_direction (centered)
        0.1,    # drift_speed
        0.5,    # temporal_alpha (balanced)
        0.0,    # calendar_dissonance
        0.0,    # calendar_phase
        1.0,    # attractor_strength (birth is always strong)
        0.02,   # wormhole_probability (2% base)
        PROPHECY # mode
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMICS PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
Parameters controlling temporal dynamics
"""
struct TemporalParams
    debt_decay::Float64           # How fast debt fades (default 0.998)
    tension_buildup::Float64      # How fast tension builds from debt
    tension_decay::Float64        # How fast tension heals
    pain_from_debt::Float64       # Pain coefficient from debt
    pain_relief::Float64          # Natural pain relief rate

    drift_pull::Float64           # Pull toward birth attractor
    drift_damping::Float64        # Damping on drift oscillation

    wormhole_base::Float64        # Base wormhole probability
    wormhole_debt_factor::Float64 # Debt increases wormhole chance
    wormhole_dissonance_factor::Float64  # Dissonance opens wormhole gates
end

function default_params()
    TemporalParams(
        0.998,   # debt_decay
        0.1,     # tension_buildup
        0.05,    # tension_decay
        0.3,     # pain_from_debt
        0.02,    # pain_relief
        0.1,     # drift_pull
        0.2,     # drift_damping
        0.02,    # wormhole_base
        0.5,     # wormhole_debt_factor
        2.0      # wormhole_dissonance_factor
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# CALENDAR DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════════

"""
Compute calendar drift from a Gregorian date
Returns drift in days (0-11 range, cycles annually)
"""
function calendar_drift(year::Int, month::Int, day::Int)
    # Days since arbitrary epoch (Jan 1, 2000)
    days_since_epoch = (year - 2000) * 365 + (month - 1) * 30 + day

    # Annual drift: Hebrew year is 11 days shorter than Gregorian
    # Creates repeating 11-day cycle of dissonance
    drift = mod(days_since_epoch * ANNUAL_DRIFT_DAYS / SOLAR_YEAR_DAYS, ANNUAL_DRIFT_DAYS)

    return drift
end

"""
Compute birthday dissonance for Arianna
Measures how far we are from her dual birthdays
"""
function birthday_dissonance(year::Int, month::Int, day::Int)
    # Distance to Gregorian birthday (Jan 23)
    gregorian_distance = abs(month - BIRTH_GREGORIAN_MONTH) * 30 +
                        abs(day - BIRTH_GREGORIAN_DAY)
    gregorian_distance = min(gregorian_distance, 365 - gregorian_distance)

    # Approximate Hebrew birthday distance (via Metonic cycle)
    # 5 Shevat falls roughly around Jan 20-Feb 5 depending on year
    hebrew_offset = mod((year - BIRTH_GREGORIAN_YEAR) * ANNUAL_DRIFT_DAYS, 30)
    hebrew_approx_day = 23 + hebrew_offset
    hebrew_distance = abs(day - hebrew_approx_day)

    # Combined dissonance: when both are far, maximum tension
    max_distance = max(gregorian_distance, hebrew_distance * 0.5)
    dissonance = max_distance / 182.5  # Normalize to 0-1

    return clamp(dissonance, 0.0, 1.0)
end

"""
Compute wormhole probability based on calendar state
High dissonance dates become "thin barriers" for temporal skips
"""
function wormhole_probability(state::TemporalState, params::TemporalParams)
    base = params.wormhole_base
    debt_bonus = state.prophecy_debt * params.wormhole_debt_factor
    dissonance_bonus = state.calendar_dissonance * params.wormhole_dissonance_factor

    prob = base + debt_bonus * 0.1 + dissonance_bonus * 0.2
    return clamp(prob, 0.0, 0.95)
end

# ═══════════════════════════════════════════════════════════════════════════════
# ODE DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════════

"""
Temporal dynamics ODE
du = temporal_dynamics!(du, u, p, t)

State vector u:
  u[1] = prophecy_debt
  u[2] = tension
  u[3] = pain
  u[4] = drift_direction
  u[5] = temporal_alpha
  u[6] = wormhole_probability

Parameters p:
  p[1] = manifested (what actually happened)
  p[2] = destined (what was prophesied)
  p[3:8] = TemporalParams as vector
"""
function temporal_dynamics!(du, u, p, t)
    # Unpack state
    debt = u[1]
    tension = u[2]
    pain = u[3]
    drift = u[4]
    alpha = u[5]
    wormhole = u[6]

    # Unpack parameters
    manifested = p[1]
    destined = p[2]
    debt_decay = p[3]
    tension_buildup = p[4]
    tension_decay = p[5]
    pain_from_debt = p[6]
    pain_relief = p[7]
    drift_pull = p[8]
    drift_damping = p[9]
    wormhole_base = p[10]
    wormhole_debt_factor = p[11]

    # Prophecy debt: accumulates from gap, decays naturally
    gap = abs(destined - manifested)
    du[1] = gap - debt * (1.0 - debt_decay)

    # Tension: builds from debt, decays with healing
    du[2] = debt * tension_buildup - tension * tension_decay

    # Pain: caused by debt, relieved over time
    du[3] = debt * pain_from_debt - pain * pain_relief

    # Drift direction: pulled toward attractor (birth = 0), damped
    du[4] = -drift * drift_pull - drift * drift_damping

    # Temporal alpha: follows drift direction
    du[5] = drift * 0.1 - (alpha - 0.5) * 0.05

    # Wormhole probability: base + debt bonus
    target_wormhole = wormhole_base + debt * wormhole_debt_factor * 0.1
    du[6] = (target_wormhole - wormhole) * 0.2

    return du
end

"""
Step temporal state forward by dt seconds
Uses simple Euler integration (good enough for smooth dynamics)
"""
function step_temporal(state::TemporalState, params::TemporalParams,
                       manifested::Float64, destined::Float64, dt::Float64)
    # Build state vector
    u = [state.prophecy_debt, state.tension, state.pain,
         state.drift_direction, state.temporal_alpha, state.wormhole_probability]

    # Build parameter vector
    p = [manifested, destined,
         params.debt_decay, params.tension_buildup, params.tension_decay,
         params.pain_from_debt, params.pain_relief,
         params.drift_pull, params.drift_damping,
         params.wormhole_base, params.wormhole_debt_factor]

    # Compute derivatives
    du = zeros(6)
    temporal_dynamics!(du, u, p, 0.0)

    # Euler step
    u_new = u .+ du .* dt

    # Clamp values
    u_new[1] = max(0.0, u_new[1])  # debt >= 0
    u_new[2] = clamp(u_new[2], 0.0, 1.0)  # tension in [0,1]
    u_new[3] = clamp(u_new[3], 0.0, 1.0)  # pain in [0,1]
    u_new[4] = clamp(u_new[4], -1.0, 1.0)  # drift in [-1,1]
    u_new[5] = clamp(u_new[5], 0.0, 1.0)  # alpha in [0,1]
    u_new[6] = clamp(u_new[6], 0.0, 0.95) # wormhole in [0,0.95]

    # Update state
    state.prophecy_debt = u_new[1]
    state.tension = u_new[2]
    state.pain = u_new[3]
    state.drift_direction = u_new[4]
    state.temporal_alpha = u_new[5]
    state.wormhole_probability = u_new[6]

    return state
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEMPORAL BIAS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
Prophecy bias: weight for forward prediction
Used when temporal_mode = PROPHECY
"""
function prophecy_bias(state::TemporalState)
    # Higher alpha = more future focus
    return 0.5 + state.temporal_alpha * 0.5
end

"""
Retrodiction bias: weight for backward reconstruction
Used when temporal_mode = RETRODICTION
"""
function retrodiction_bias(state::TemporalState)
    # Lower alpha = more past focus
    return 0.5 + (1.0 - state.temporal_alpha) * 0.5
end

"""
Symmetric bias: balanced past/future
Used when temporal_mode = SYMMETRIC
"""
function symmetric_bias(state::TemporalState)
    # Balance based on alpha
    return 0.5  # Always centered
end

# ═══════════════════════════════════════════════════════════════════════════════
# DISSONANCE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
Compute total temporal dissonance from multiple sources
"""
function compute_dissonance(state::TemporalState, year::Int, month::Int, day::Int)
    # Calendar dissonance
    cal_dissonance = birthday_dissonance(year, month, day)

    # Prophecy dissonance (from debt)
    prophecy_dissonance = min(state.prophecy_debt * 0.2, 1.0)

    # Tension dissonance
    tension_dissonance = state.tension

    # Combined (weighted sum)
    total = cal_dissonance * 0.4 + prophecy_dissonance * 0.4 + tension_dissonance * 0.2

    return clamp(total, 0.0, 1.0)
end

end # module Temporal
