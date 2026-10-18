"""
Отладочная отрисовка и телеметрия стратегии (ТЗ, раздел 14)
"""

from bridge import drawing
from bridge.auxiliary import aux
from bridge.strategy import params

GREEN = (0, 220, 0)
RED = (230, 0, 0)
YELLOW = (240, 220, 0)
WHITE = (255, 255, 255)
BLUE = (60, 140, 255)
ORANGE = (255, 150, 0)
GRAY = (150, 150, 150)
CYAN = (0, 220, 220)


def p_color(p: float) -> tuple[int, int, int]:
    """Цвет по вероятности: красный -> зелёный"""
    p = max(0.0, min(1.0, p))
    return int(230 * (1 - p)), int(220 * p), 0


class DebugDraw:
    """Обёртка над strategy_image с лимитом примитивов за такт"""

    def __init__(self, image: drawing.Image, enabled: bool = params.DEBUG_DRAW) -> None:
        self.image = image
        self.enabled = enabled
        self.count = 0
        self.telemetry: list[str] = []

    def _allow(self) -> bool:
        if not self.enabled or self.count >= params.DEBUG_DRAW_MAX_PRIMITIVES:
            return False
        self.count += 1
        return True

    def circle(self, pos: aux.Point, color: tuple[int, int, int] = WHITE, radius: float = 30) -> None:
        """Круг"""
        if self._allow():
            self.image.draw_circle(pos, color, radius)

    def line(self, a: aux.Point, b: aux.Point, color: tuple[int, int, int] = WHITE, width: int = 3) -> None:
        """Линия"""
        if self._allow():
            self.image.draw_line(a, b, color, width)

    def text(self, pos: aux.Point, text: str, color: tuple[int, int, int] = WHITE, size: int = 80) -> None:
        """Подпись"""
        if self._allow():
            self.image.print(pos, text, color, size)

    def poly(self, points: list[aux.Point], color: tuple[int, int, int] = WHITE) -> None:
        """Многоугольник"""
        if self._allow():
            self.image.draw_poly(points, color)

    def tele(self, line: str) -> None:
        """Строка телеметрии"""
        self.telemetry.append(line)

    def flush(self, name: str = "STRATEGY") -> None:
        """Отправить телеметрию"""
        if self.telemetry:
            self.image.send_telemetry(name, "\n".join(self.telemetry))
        self.telemetry = []
