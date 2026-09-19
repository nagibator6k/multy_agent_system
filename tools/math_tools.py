import math


def solve_quadratic(a: float, b: float, c: float) -> dict:
    """
    Solve a quadratic equation:
        ax² + bx + c = 0

    Returns:
        Dictionary containing discriminant and roots.
    """

    if a == 0:
        raise ValueError("Coefficient 'a' must not be zero.")

    discriminant = b ** 2 - 4 * a * c

    if discriminant < 0:
        return {
            "discriminant": discriminant,
            "roots": [],
            "root_count": 0,
        }

    if discriminant == 0:
        root = -b / (2 * a)

        return {
            "discriminant": discriminant,
            "roots": [root],
            "root_count": 1,
        }

    sqrt_d = math.sqrt(discriminant)

    x1 = (-b + sqrt_d) / (2 * a)
    x2 = (-b - sqrt_d) / (2 * a)

    return {
        "discriminant": discriminant,
        "roots": [x1, x2],
        "root_count": 2,
    }