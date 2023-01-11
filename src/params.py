DEFAULT_D_VALUE = 400
DEFAULT_SCORING_FUNCTION_BASE = 1.5
LOG_BASE = 10

def get_k_value(x: float) -> float:
    if x < 0:
        return 0
    elif x < 800:
        return 400
    elif x < 1200:
        return 300
    elif x < 1600:
        return 200
    elif x < 2000:
        return 100
    elif x < 2400:
        return 80
    else:
        return 50