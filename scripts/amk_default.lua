--[[
  amk_default.lua - Default AMK configuration script

  This script is hot-reloadable! Edit it while arianna is running
  and call amk_lua_reload() to apply changes.

  "The field IS the script. The script IS the field."

  Available API:
    amk.prophecy()         -> int    (current prophecy depth)
    amk.destiny()          -> float  (destiny pull 0-1)
    amk.wormhole()         -> float  (wormhole chance 0-1)
    amk.pain()             -> float  (pain level 0-1)
    amk.tension()          -> float  (tension level 0-1)
    amk.dissonance()       -> float  (dissonance 0-1)
    amk.velocity()         -> int    (0=nomove, 1=walk, 2=run, -1=backward)
    amk.effective_temp()   -> float  (current effective temperature)
    amk.debt()             -> float  (current debt)
    amk.temporal_debt()    -> float  (temporal debt from backward movement)
    amk.pack_enabled(name) -> bool   (check if pack is enabled)

    amk.set_prophecy(n)        -- set prophecy depth 1-64
    amk.set_destiny(f)         -- set destiny pull 0-1
    amk.set_wormhole(f)        -- set wormhole chance 0-1
    amk.set_pain(f)            -- set pain 0-1
    amk.set_tension(f)         -- set tension 0-1
    amk.set_dissonance(f)      -- set dissonance 0-1
    amk.set_velocity(mode)     -- "walk", "run", "nomove", "backward"
    amk.jump(n)                -- queue jump of n tokens
    amk.exec(script)           -- execute raw DSL script
    amk.reset_field()          -- reset suffering state
    amk.reset_debt()           -- reset debt
    amk.enable_pack(name)      -- enable pack by name
    amk.disable_pack(name)     -- disable pack by name
    amk.step(dt)               -- advance field physics

  Callbacks (define these functions to hook into generation):
    on_generate_start(prompt, max_tokens, temperature)
    on_generate_end(output, tokens_generated)
    on_token(token_id, prob, position)
    on_trauma(intensity, trigger)
    on_emotion(valence, arousal)
]]

-- ═══════════════════════════════════════════════════════════════════════════════
-- CONFIGURATION (edit these to change personality dynamics)
-- ═══════════════════════════════════════════════════════════════════════════════

local config = {
  -- Base field dynamics
  prophecy = 7,           -- lookahead depth (1-64)
  destiny = 0.35,         -- pull toward prophesied tokens (0-1)
  wormhole_base = 0.12,   -- base wormhole chance (0-1)

  -- Response patterns
  trauma_wormhole_boost = 0.15,   -- extra wormhole chance when traumatized
  high_arousal_threshold = 0.7,   -- arousal level that triggers intensity
  pain_velocity_slowdown = true,  -- reduce velocity when in pain

  -- Pack defaults
  enable_codes_ric = false,
  enable_darkmatter = false,

  -- Velocity patterns
  default_velocity = "walk",
}

-- ═══════════════════════════════════════════════════════════════════════════════
-- INITIALIZATION (called when script loads)
-- ═══════════════════════════════════════════════════════════════════════════════

local function init()
  print("[Lua] Initializing AMK field dynamics...")

  -- Apply base configuration
  amk.set_prophecy(config.prophecy)
  amk.set_destiny(config.destiny)
  amk.set_wormhole(config.wormhole_base)
  amk.set_velocity(config.default_velocity)

  -- Enable packs if configured
  if config.enable_codes_ric then
    amk.enable_pack("CODES_RIC")
  end
  if config.enable_darkmatter then
    amk.enable_pack("DARKMATTER")
  end

  print("[Lua] Field initialized: prophecy=" .. amk.prophecy() ..
        ", destiny=" .. string.format("%.2f", amk.destiny()))
end

-- ═══════════════════════════════════════════════════════════════════════════════
-- CALLBACKS
-- ═══════════════════════════════════════════════════════════════════════════════

-- Called before generation starts
function on_generate_start(prompt, max_tokens, temperature)
  print(string.format("[Lua] Generation starting: %d tokens, temp=%.2f",
                      max_tokens, temperature))

  -- Adjust prophecy based on prompt length
  local prompt_len = #prompt
  if prompt_len > 100 then
    -- Longer prompts = more context = can look further ahead
    amk.set_prophecy(math.min(config.prophecy * 2, 32))
  else
    amk.set_prophecy(config.prophecy)
  end

  -- Reset tension for fresh start
  if amk.tension() > 0.5 then
    amk.set_tension(amk.tension() * 0.5)  -- reduce by half, don't eliminate
  end
end

-- Called when generation ends
function on_generate_end(output, tokens_generated)
  print(string.format("[Lua] Generated %d tokens", tokens_generated))

  -- Step the field physics
  local dt = tokens_generated / 100.0  -- ~10ms per token
  amk.step(dt)

  -- Check if we accumulated too much debt
  if amk.debt() > 5.0 then
    print("[Lua] High debt detected, allowing decay")
    -- Debt will decay naturally via am_step
  end
end

-- Called on each token (use sparingly - performance impact)
function on_token(token_id, prob, position)
  -- Only process every 10th token to reduce overhead
  if position % 10 ~= 0 then return end

  -- Low probability tokens increase wormhole chance
  if prob < 0.1 then
    local current = amk.wormhole()
    if current < 0.5 then
      amk.set_wormhole(current + 0.02)
    end
  end
end

-- Called when trauma is detected
function on_trauma(intensity, trigger)
  print(string.format("[Lua] Trauma detected: intensity=%.2f, trigger='%s'",
                      intensity, trigger))

  if intensity > 0.5 then
    -- High trauma: boost wormhole (escape tendency)
    amk.set_wormhole(config.wormhole_base + config.trauma_wormhole_boost)

    -- Increase pain proportionally
    amk.set_pain(math.min(amk.pain() + intensity * 0.3, 1.0))

    -- Slow down if configured
    if config.pain_velocity_slowdown then
      amk.set_velocity("walk")
    end
  end
end

-- Called on emotional drift
function on_emotion(valence, arousal)
  -- Negative valence -> increase tension
  if valence < -0.3 then
    amk.set_tension(math.min(amk.tension() + 0.1, 1.0))
  end

  -- High arousal -> increase velocity
  if arousal > config.high_arousal_threshold then
    if amk.velocity() < amk.VEL_RUN then
      amk.set_velocity("run")
    end
  elseif arousal < 0.3 then
    -- Low arousal -> slow down
    amk.set_velocity("walk")
  end
end

-- ═══════════════════════════════════════════════════════════════════════════════
-- UTILITY FUNCTIONS (available to other scripts)
-- ═══════════════════════════════════════════════════════════════════════════════

-- Gradually adjust a parameter over time
function ease_toward(current, target, rate)
  rate = rate or 0.1
  return current + (target - current) * rate
end

-- Apply a mood preset
function apply_mood(mood_name)
  local moods = {
    calm = {prophecy = 5, destiny = 0.2, wormhole = 0.05, velocity = "walk"},
    intense = {prophecy = 12, destiny = 0.6, wormhole = 0.2, velocity = "run"},
    contemplative = {prophecy = 20, destiny = 0.4, wormhole = 0.1, velocity = "nomove"},
    chaotic = {prophecy = 3, destiny = 0.1, wormhole = 0.4, velocity = "run"},
    nostalgic = {prophecy = 15, destiny = 0.5, wormhole = 0.08, velocity = "backward"},
  }

  local preset = moods[mood_name]
  if preset then
    amk.set_prophecy(preset.prophecy)
    amk.set_destiny(preset.destiny)
    amk.set_wormhole(preset.wormhole)
    amk.set_velocity(preset.velocity)
    print("[Lua] Applied mood: " .. mood_name)
  else
    print("[Lua] Unknown mood: " .. mood_name)
  end
end

-- ═══════════════════════════════════════════════════════════════════════════════
-- RUN INITIALIZATION
-- ═══════════════════════════════════════════════════════════════════════════════

init()

print("[Lua] AMK script loaded. Edit and reload at runtime!")
print("[Lua] Try: apply_mood('calm'), apply_mood('intense'), apply_mood('chaotic')")
