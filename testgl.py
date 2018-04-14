from GL import GL
import numpy as np


def main():
    gl = GL(1280, 1024)

    loc = np.random.random_sample([10, 3]) - 0.5
    col = np.random.random_sample([10, 4])
    rad = np.random.random_sample([10])*0.1

    def display():
        for i in range(10):
            gl.gl_draw_sphere(loc[i], rad[i], col[i])

    gl._flag_draw_cells = True
    gl._flag_draw_axes = True
    gl.gl_set_display_function(display)
    gl.gl_run()


main()
