from src.flappybird import config

INIT_PIPE_COOLDOWN = config.INIT_PIPE_COOLDOWN
MIN_PIPE_COOLDOWN = config.MIN_PIPE_COOLDOWN
SLOW_PHASE_COOLDOWN = config.SLOW_PHASE_COOLDOWN
FAST_SCORE_STEP = config.FAST_SCORE_STEP
FAST_COOLDOWN_CHANGE = config.FAST_COOLDOWN_CHANGE
SLOW_SCORE_STEP = config.SLOW_SCORE_STEP
SLOW_COOLDOWN_CHANGE = config.SLOW_COOLDOWN_CHANGE

# score at which the fast phase ends and the cooldown starts shrinking slower
FAST_PHASE_STEPS = (INIT_PIPE_COOLDOWN - SLOW_PHASE_COOLDOWN) // FAST_COOLDOWN_CHANGE
FAST_PHASE_END_SCORE = FAST_PHASE_STEPS * FAST_SCORE_STEP

# score at which the cooldown hits MIN_PIPE_COOLDOWN (hardest difficulty)
SLOW_PHASE_STEPS = -(-(SLOW_PHASE_COOLDOWN - MIN_PIPE_COOLDOWN) // SLOW_COOLDOWN_CHANGE)
MAX_DIFFICULTY_SCORE = FAST_PHASE_END_SCORE + SLOW_PHASE_STEPS * SLOW_SCORE_STEP

def get_pipe_cooldown(score):
    # fast phase: cooldown drops quickly while the game is still easy
    if score < FAST_PHASE_END_SCORE:
        return INIT_PIPE_COOLDOWN - (score // FAST_SCORE_STEP) * FAST_COOLDOWN_CHANGE

    # slow phase: cooldown keeps dropping, but less often and by less
    pipe_cd = INIT_PIPE_COOLDOWN - FAST_PHASE_STEPS * FAST_COOLDOWN_CHANGE
    pipe_cd -= ((score - FAST_PHASE_END_SCORE) // SLOW_SCORE_STEP) * SLOW_COOLDOWN_CHANGE
    return max(pipe_cd, MIN_PIPE_COOLDOWN)
