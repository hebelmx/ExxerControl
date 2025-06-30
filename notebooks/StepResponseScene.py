
from manim import *

from manim import *

class StepResponseScene(Scene):
    def construct(self):
        title = Title(r"$\text{Signal Propagation: } u_1 \to y_1$")
        self.play(Write(title))

        u1 = MathTex("u_1").to_edge(LEFT).shift(UP * 1)
        y1 = MathTex("y_1").to_edge(RIGHT).shift(UP * 1)
        G11 = MathTex("G_{11}(s)").move_to(ORIGIN)

        self.play(FadeIn(u1), FadeIn(y1), FadeIn(G11))

        arrow1 = Arrow(u1.get_right(), G11.get_left(), buff=0.1)
        arrow2 = Arrow(G11.get_right(), y1.get_left(), buff=0.1)

        self.play(GrowArrow(arrow1), GrowArrow(arrow2))

        # Signal pulse: simulate signal traveling from u1 to y1
        pulse = Dot(color=YELLOW).move_to(u1.get_right())
        self.play(FadeIn(pulse))
        self.play(pulse.animate.move_to(G11.get_left()), run_time=1)
        self.play(pulse.animate.move_to(G11.get_right()), run_time=1)
        self.play(pulse.animate.move_to(y1.get_left()), run_time=1)
        self.play(FadeOut(pulse))

        self.wait(2)

