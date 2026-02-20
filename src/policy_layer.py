"""
Optional policy layer: map model IPO weight (or tilt) to a rule for
how much to participate in the next IPO (e.g. scale position size).
"""


def ipo_tilt_to_position_scale(ipo_weight: float, scale_max: float = 1.0) -> float:
    """
    Map current IPO sleeve weight to a position scale for the next IPO.

    When ipo_weight is high, scale is high (participate more).
    Returns value in [0, scale_max].
    """
    return min(scale_max, max(0.0, ipo_weight * (1.0 / 0.5)))  # 50% IPO weight -> scale 1.0


def policy_rule(ipo_weight: float, threshold: float = 0.2) -> str:
    """
    Simple rule for retail: "consider increasing IPO exposure" when tilt is above threshold.
    """
    if ipo_weight > threshold:
        return "Consider increasing IPO exposure (model IPO weight > {:.0%}).".format(threshold)
    return "Model suggests moderate or low IPO allocation."
