from manim import *

class MimoSystemScene(Scene):
    def construct(self):
        title = Title("3x3 MIMO Control Structure", include_underline=True)
        self.play(Write(title))

        inputs = [MathTex(f"u_{i+1}") for i in range(3)]
        outputs = [MathTex(f"y_{i+1}") for i in range(3)]
        blocks = [[MathTex(f"G_{{{i+1}{j+1}}}(s)") for j in range(3)] for i in range(3)]

        for i, inp in enumerate(inputs):
            inp.to_edge(LEFT).shift(DOWN * (i - 1))
        for j, out in enumerate(outputs):
            out.to_edge(RIGHT).shift(DOWN * (j - 1))
        for i in range(3):
            for j in range(3):
                blocks[i][j].move_to((j - 1) * RIGHT + (1 - i) * UP)

        for obj in inputs + outputs:
            self.play(FadeIn(obj))
        for row in blocks:
            for blk in row:
                self.play(FadeIn(blk, shift=UP))

        for i in range(3):
            for j in range(3):
                arrow = Arrow(inputs[j].get_right(), blocks[i][j].get_left(), buff=0.1)
                self.play(GrowArrow(arrow), run_time=0.2)
                arrow2 = Arrow(blocks[i][j].get_right(), outputs[i].get_left(), buff=0.1)
                self.play(GrowArrow(arrow2), run_time=0.2)

        self.wait(2)
