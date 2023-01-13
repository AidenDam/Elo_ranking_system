DEFAULT_D_VALUE = 400
DEFAULT_SCORING_FUNCTION_BASE = 1.5
LOG_BASE = 10

def get_k_value(x: float) -> float:
    ks = [0, 16, 32, 64, 128, 256, 512]
    x_t = abs(x)
    if x_t < 0:
        i = 0
    elif x_t < 800:
        i = 1
    elif x_t < 1200:
        i = 2
    elif x_t < 1600:
        i = 3
    elif x_t < 2000:
        i = 4
    elif x_t < 2400:
        i = 5
    else:
        i = 6
    return ks[-i if x < 0 else i]