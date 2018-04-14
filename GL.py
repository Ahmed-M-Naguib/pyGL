
# developed by Naguib
# 2018 April 12th

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
from math import *


class GL(object):
    def __init__(self, width, height):
        self._draw_scale = 50.0
        self._camera = np.array([-150.0, 150.0, 50.0])
        self._target = np.zeros([3])
        self._phantom = np.zeros([3])
        self._flag_mouse_left_down = False
        self._flag_mouse_right_down = False
        self._flag_keyboard_shift_down = False
        self._flag_keyboard_ctrl_down = False
        self._flag_draw_anchor = False
        self._flag_draw_axes = False
        self._flag_draw_cells = False
        self._flag_lock_target = False
        self._bg_color = np.array([1.0, 1.0, 1.0])
        self._width = width
        self._height = height
        self._viewport = np.zeros([4])
        self._modelview = np.zeros([4, 4])
        self._projection = np.zeros([4, 4])
        self._mouse_down = np.zeros([2])
        self._camera_angles = np.zeros([2])
        self._light_position = np.array([-100.0, -100.0, 100.0, 1.0])
        self._callback_display_function = None
        self._callback_resize_function = None
        self._callback_keyboard_function = None
        self._callback_keyboard_special_function = None
        self._callback_mouse_click_function = None
        self._callback_mouse_move_function = None
        self._callback_timer_function = None
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(self._width, self._height)
        glutInitWindowPosition(0, 0)
        self._windowID = glutCreateWindow("GL v1.0")
        glutDisplayFunc(self._gl_display)
        glutIdleFunc(self._gl_display)
        glutReshapeFunc(self._gl_resize)
        glutKeyboardFunc(self._gl_keyboard)
        glutSpecialFunc(self._gl_keyboard_special)
        glutMouseFunc(self._gl_mouse_click)
        glutMotionFunc(self._gl_mouse_move)
        glutTimerFunc(20, self._gl_timer, 0)
        glClearColor(self._bg_color[0], self._bg_color[1], self._bg_color[2], 1.0)
        glViewport(0, 0, self._width, self._height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, self._width / self._height, 0.1, 100000.0)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    @staticmethod
    def _gl_color(color):
        color = color.flatten()
        if len(color) == 4:
            glColor4f(*color)
        elif len(color) == 3:
            glColor3f(*color)

    def _gl_display(self):
        glViewport(0, 0, self._width, self._height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, self._width / self._height, 0.1, 100000.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(self._camera[0], self._camera[1], self._camera[2],
                  self._target[0], self._target[1], self._target[2], 0.0, 0.0, 1.0)
        glGetDoublev(GL_MODELVIEW_MATRIX, self._modelview)
        glGetDoublev(GL_PROJECTION_MATRIX, self._projection)
        glGetIntegerv(GL_VIEWPORT, self._viewport)
        glDisable(GL_LIGHTING)
        if self._flag_draw_axes:
            self._gl_draw_axes()
        if self._flag_draw_cells:
            self._gl_draw_cells()
        if self._callback_display_function:
            self._callback_display_function()
        glLightfv(GL_LIGHT1, GL_POSITION, self._light_position)
        glutSwapBuffers()

    def gl_draw_text(self, msg, location, color, font):
        font_dict = {'8': 'GLUT_BITMAP_8_BY_13',
                     '9': 'GLUT_BITMAP_9_BY_15',
                     '10_times': 'GLUT_BITMAP_TIMES_ROMAN_10',
                     '24_times': 'GLUT_BITMAP_TIMES_ROMAN_24',
                     '10_helvetica': 'GLUT_BITMAP_HELVETICA_10',
                     '12_helvetica': 'GLUT_BITMAP_HELVETICA_12',
                     '18_helvetica': 'GLUT_BITMAP_HELVETICA_18'}

        c_font = platform.getGLUTFontPointer(font_dict[font])

        location = location.flatten()
        self._gl_color(color)

        if len(location) == 3:
            glRasterPos3f(*(location*self._draw_scale))
        elif len(location) == 2:
            glPushMatrix()
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glOrtho(0.0, self._width, self._height, 0.0, -1.0, 10.0)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glRasterPos2f(*location)

        for c in msg:
            glutBitmapCharacter(c_font, ctypes.c_int(ord(c)))

        if len(location) == 2:
            glMatrixMode(GL_PROJECTION)
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()

    def gl_draw_octree(self, cells, color, thickness):
        for c in cells:
            if c.occlusion == 0:
                self.gl_draw_box(c.location, c.cell_size, color, thickness)
            else:
                self.gl_draw_box(c.location, c.cell_size, color, thickness)
                self.gl_draw_line(c.location, c.occluding_cell.location, color, thickness)

    def gl_lock_target(self):
        self._flag_lock_target = True

    def gl_unlock_target(self):
        self._flag_lock_target = False

    def gl_draw_box(self, location, size, color, thickness):
        location = location.flatten() * self._draw_scale
        size = size.flatten() * self._draw_scale

        self.gl_draw_line(location + 0.5*size*[-1, -1, -1], location + 0.5*size*[+1, -1, -1], color, thickness)
        self.gl_draw_line(location + 0.5*size*[-1, +1, -1], location + 0.5*size*[+1, +1, -1], color, thickness)
        self.gl_draw_line(location + 0.5*size*[-1, -1, -1], location + 0.5*size*[-1, +1, -1], color, thickness)
        self.gl_draw_line(location + 0.5*size*[+1, -1, -1], location + 0.5*size*[+1, +1, -1], color, thickness)

        self.gl_draw_line(location + 0.5*size*[-1, -1, +1], location + 0.5*size*[+1, -1, +1], color, thickness)
        self.gl_draw_line(location + 0.5*size*[-1, +1, +1], location + 0.5*size*[+1, +1, +1], color, thickness)
        self.gl_draw_line(location + 0.5*size*[-1, -1, +1], location + 0.5*size*[-1, +1, +1], color, thickness)
        self.gl_draw_line(location + 0.5*size*[+1, -1, +1], location + 0.5*size*[+1, +1, +1], color, thickness)

        self.gl_draw_line(location + 0.5*size*[-1, -1, -1], location + 0.5*size*[-1, -1, +1], color, thickness)
        self.gl_draw_line(location + 0.5*size*[+1, -1, -1], location + 0.5*size*[+1, -1, +1], color, thickness)
        self.gl_draw_line(location + 0.5*size*[-1, +1, -1], location + 0.5*size*[-1, +1, +1], color, thickness)
        self.gl_draw_line(location + 0.5*size*[+1, +1, -1], location + 0.5*size*[+1, +1, +1], color, thickness)

    def gl_set_display_function(self, display_function):
        self._callback_display_function = display_function

    def gl_set_resize_function(self, resize_function):
        self._callback_resize_function = resize_function

    def gl_set_keyboard_function(self, keyboard_function):
        self._callback_keyboard_function = keyboard_function

    def gl_set_keyboard_special_function(self, keyboard_special_function):
        self._callback_keyboard_special_function = keyboard_special_function

    def gl_set_mouse_click_function(self, mouse_click_function):
        self._callback_mouse_click_function = mouse_click_function

    def gl_set_mouse_move_function(self, mouse_move_function):
        self._callback_mouse_move_function = mouse_move_function

    def gl_set_timer_function(self, timer_function):
        self._callback_timer_function = timer_function

    def gl_draw_scale(self, val=None):
        if val:
            self._draw_scale = val
        return self._draw_scale

    def gl_camera(self, val=None):
        if val:
            if len(val.flatten()) == 3:
                self._camera = val.flatten()
        return self._camera

    def gl_target(self, val=None):
        if val:
            if len(val.flatten()) == 3:
                self._target = val.flatten()
        return self._target

    def _gl_raytrace_2d_point(self, uv):
        if self._viewport[0] <= uv[0] < self._viewport[2] and self._viewport[1] <= uv[1] < self._viewport[3]:
            depth = 0
            depth = glReadPixels(uv[0], self._viewport[3] - uv[1]*1.0, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, depth)
            point3d = gluUnProject(uv[0], self._viewport[3] - uv[1]*1.0, depth, self._modelview, self._projection, self._viewport)
            if point3d.abs().max() == 0 or point3d.abs().max() > 10000:
                return np.zeros([3])
            return point3d / self._draw_scale
        else:
            return np.zeros([3])

    def _gl_resize(self, new_width, new_height):
        self._width = new_width
        self._height = new_height
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(0, 0, self._width, self._height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, self._width / self._height, 5.0, 10000.0)
        glutPostRedisplay()
        if self._callback_resize_function:
            self._callback_resize_function(new_width, new_height)

    def _gl_keyboard(self, p_key, p_x, p_y):
        if p_key == b'\x1b':
            exit(1)
        elif p_key == b'0':
            self._phantom = self._target = np.zeros([3])
        if self._callback_keyboard_function:
            self._callback_keyboard_function(p_key, p_x, p_y)

    def _gl_keyboard_special(self, p_key, p_x, p_y):
        if self._callback_keyboard_special_function:
            self._callback_keyboard_special_function(p_key, p_x, p_y)

    def _gl_mouse_click(self, button, state, x, y):

        p3d = self._gl_raytrace_2d_point([x, y])

        if self._callback_mouse_click_function:
            self._callback_mouse_click_function(button, state, x, y, p3d, self._flag_mouse_left_down, self._flag_mouse_right_down, self._flag_keyboard_shift_down, self._flag_keyboard_ctrl_down)
        special_key = glutGetModifiers()
        if button == GLUT_LEFT_BUTTON:
            self._flag_mouse_left_down = True
        elif button == GLUT_RIGHT_BUTTON:
            direction = self._camera-self._target
            direction /= np.linalg.norm(direction)
            directionxy = direction
            directionxy[2] = 0
            self._camera_angles[1] = atan2(sqrt(directionxy.dot(directionxy)), direction[2])
            self._camera_angles[0] = atan2(direction[1], direction[0])
            self._flag_mouse_right_down = True
        self._mouse_down = np.array([x, y])

        if state == GLUT_UP:
            if self._flag_keyboard_ctrl_down and self._flag_mouse_left_down and not self._flag_mouse_right_down and np.max(np.abs(p3d)) > 0:
                self._phantom = p3d
                self._target = p3d
            if button == GLUT_LEFT_BUTTON:
                self._flag_mouse_left_down = False
            if button == GLUT_RIGHT_BUTTON:
                self._flag_mouse_right_down = False

        if special_key == GLUT_ACTIVE_SHIFT:
            self._flag_keyboard_shift_down = True
        else:
            self._flag_keyboard_shift_down = False
        if special_key == GLUT_ACTIVE_CTRL:
            self._flag_keyboard_ctrl_down = True
        else:
            self._flag_keyboard_ctrl_down = False

        self._flag_draw_anchor = self._flag_keyboard_shift_down or self._flag_keyboard_ctrl_down

    def _gl_mouse_move(self, x, y):
        p3d = self._gl_raytrace_2d_point([x, y])
        step = self._mouse_down - [x, y]
        direction = self._target - self._camera
        norm_direction = np.linalg.norm(direction)
        direction /= norm_direction

        if self._callback_mouse_move_function:
            self._callback_mouse_move_function(x, y, self._mouse_down, p3d, self._flag_mouse_left_down, self._flag_mouse_right_down, self._flag_keyboard_shift_down, self._flag_keyboard_ctrl_down)

        if self._flag_mouse_left_down and self._flag_mouse_right_down:
            if self._flag_keyboard_shift_down or self._flag_lock_target:
                self._camera += direction * step[1]
                self._phantom = self._target
            else:
                self._target += direction * step[1]
                self._camera += direction * step[1]
                self._phantom = self._target

        elif self._flag_mouse_right_down:
            rotation = step * 0.01
            self._camera_angles[0] += rotation[0]
            self._camera_angles[1] += rotation[1]
            dif = np.array([norm_direction * cos(self._camera_angles[0]) * sin(self._camera_angles[1]),
                            norm_direction * sin(self._camera_angles[0]) * sin(self._camera_angles[1]),
                            norm_direction * cos(self._camera_angles[1])])
            if self._flag_keyboard_shift_down or self._flag_lock_target:
                self._camera = self._target + dif
            else:
                self._target = self._camera - dif
            self._phantom = self._target
        elif self._flag_mouse_left_down:
            if self._flag_keyboard_ctrl_down:
                cd = np.linalg.norm(self._camera / self._draw_scale - p3d)
                if cd > 0.005 and np.max(np.abs(p3d)) > 0:
                    self._phantom = p3d
            elif not (self._flag_keyboard_shift_down or self._flag_lock_target):
                cx = np.cross([0, 0, 1], direction)
                cy = np.cross(direction, cx)
                cw = cx*step[0] + cy*step[1]
                self._camera += cw
                self._target += cw
                self._phantom = self._target
        self._mouse_down = np.array([x, y])
        glutPostRedisplay()

    def gl_draw_vector(self, location, direction, length, color, thickness):
        self.gl_draw_line(location, location + direction*length, color, thickness)

    def gl_draw_line(self, location1, location2, color, thickness):
        glLineWidth(thickness)
        self._gl_color(color)
        location1 *= self._draw_scale
        location2 *= self._draw_scale
        glBegin(GL_LINES)
        glVertex3f(*location1)
        glVertex3f(*location2)
        glEnd()

    def gl_draw_sphere(self, center, radius, color, hslices=10, vslices=10):
        self._gl_color(color)
        _center = center * self._draw_scale
        _radius = radius * self._draw_scale
        glPushMatrix()
        glTranslatef(*_center)
        glutSolidSphere(_radius, hslices, vslices)
        glPopMatrix()

    def gl_draw_plane(self, n, d, width, height, color):
        n /= np.linalg.norm(n)
        if np.abs(np.abs(n.dot([1, 0, 0])) - 1) > 0.01:
            x = np.cross(n, [1, 0, 0])
        elif np.abs(np.abs(n.dot([0, 1, 0])) - 1) > 0.01:
            x = np.cross(n, [0, 1, 0])
        else:
            x = np.cross(n, [0, 0, 1])

        y = np.cross(n, x)

        if np.abs(n[0]) > np.abs(n[1]) and np.abs(n[0]) > np.abs(n[2]):
            center = np.array([-d / n[0], 0, 0])
        elif np.abs(n[1]) > np.abs(n[2]):
            center = np.array([0, -d / n[2], 0])
        else:
            center = np.array([0, 0, -d / n[2]])

        c1 = center + x * width + y * height
        c2 = center + x * width - y * height
        c3 = center - x * width - y * height
        c4 = center - x * width + y * height

        c1 *= self._draw_scale
        c2 *= self._draw_scale
        c3 *= self._draw_scale
        c4 *= self._draw_scale

        self._gl_color(color)
        glBegin(GL_POLYGON)
        glVertex3f(*c1)
        glVertex3f(*c2)
        glVertex3f(*c3)
        glVertex3f(*c4)
        glEnd()
        glBegin(GL_POLYGON)
        glVertex3f(*c1)
        glVertex3f(*c4)
        glVertex3f(*c3)
        glVertex3f(*c2)
        glEnd()

    def _gl_draw_axes(self):
        glLineWidth(3.0)
        glBegin(GL_LINES)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(990, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()
        glBegin(GL_LINES)
        glColor3f(0.5, 0.0, 0.0)
        glVertex3f(-990, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()
        glBegin(GL_LINES)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 990, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()
        glBegin(GL_LINES)
        glColor3f(0.0, 0.5, 0.0)
        glVertex3f(0.0, -990, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()
        glBegin(GL_LINES)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 990)
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()
        glBegin(GL_LINES)
        glColor3f(0.0, 0.0, 0.5)
        glVertex3f(0.0, 0.0, -990)
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()
        if self._flag_draw_anchor:
            glLineWidth(1.0)
            glBegin(GL_LINES)
            glColor3f(0.0, 1.0, 0.0)
            glVertex3f(self._phantom[0] - 5.0, self._phantom[1], self._phantom[2])
            glVertex3f(self._phantom[0] + 5.0, self._phantom[1], self._phantom[2])
            glEnd()
            glBegin(GL_LINES)
            glColor3f(0.0, 0.0, 1.0)
            glVertex3f(self._phantom[0], self._phantom[1] - 5.0, self._phantom[2])
            glVertex3f(self._phantom[0], self._phantom[1] + 5.0, self._phantom[2])
            glEnd()
            glBegin(GL_LINES)
            glColor3f(1.0, 0.0, 0.0)
            glVertex3f(self._phantom[0], self._phantom[1], self._phantom[2] - 5.0)
            glVertex3f(self._phantom[0], self._phantom[1], self._phantom[2] + 5.0)
            glEnd()

    def _gl_draw_cells(self):
        if self._width > 0:
            for i in range(-990, 990, 30):
                glLineWidth(1.0)
                glColor3f(0.5, 0.5, 0.5)
                glBegin(GL_LINES)
                glVertex3f(1000.0, i, 0.0)
                glVertex3f(-1000.0, i, 0.0)
                glEnd()
                glBegin(GL_LINES)
                glVertex3f(i, 1000.0, 0.0)
                glVertex3f(i, -1000.0, 0.0)
                glEnd()

    def gl_draw_point_cloud(self, cloud_uvxyzrgb, step=1, transform=None):
        glPointSize(2.0)
        glBegin(GL_POINTS)
        if transform:
            for i in range(0, len(cloud_uvxyzrgb), step):
                glColor3f(cloud_uvxyzrgb[i, 5] / 256.0, cloud_uvxyzrgb[i, 6] / 256.0, cloud_uvxyzrgb[i, 7] / 256.0)
                point = transform.dot(np.array([cloud_uvxyzrgb[i, 2] * self._draw_scale, cloud_uvxyzrgb[i, 3] * self._draw_scale, cloud_uvxyzrgb[i, 4] * self._draw_scale, 1.0]))
                glVertex3f(*point)
        else:
            for i in range(0, len(cloud_uvxyzrgb), step):
                glColor3f(cloud_uvxyzrgb[i, 5] / 256.0, cloud_uvxyzrgb[i, 6] / 256.0, cloud_uvxyzrgb[i, 7] / 256.0)
                glVertex3f(cloud_uvxyzrgb[i, 2]*self._draw_scale, cloud_uvxyzrgb[i, 3]*self._draw_scale, cloud_uvxyzrgb[i, 4]*self._draw_scale)
        glEnd()

    def gl_draw_image(self, image, scale=1.0):
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=2)
        assert len(image.shape) == 3
        glPushMatrix()
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0.0, self._width, self._height, 0.0, -1.0, 10.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        for _u in np.arange(0, image.shape[0], 1.0/scale):
            for _v in np.arange(0, image.shape[1], 1.0/scale):
                u = np.int32(_u)
                v = np.int32(_v)
                glBegin(GL_QUADS)
                glColor3f(*image[u, v] / 256.0)
                glVertex2f(self._width - image.shape[0]*scale + u*scale,     self._height - image.shape[1]*scale + v*scale)
                glVertex2f(self._width - image.shape[0]*scale + u*scale + 1, self._height - image.shape[1]*scale + v*scale)
                glVertex2f(self._width - image.shape[0]*scale + u*scale + 1, self._height - image.shape[1]*scale + v*scale + 1)
                glVertex2f(self._width - image.shape[0]*scale + u*scale,     self._height - image.shape[1]*scale + v*scale + 1)
                glEnd()
        glMatrixMode(GL_PROJECTION)
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def _gl_timer(self, pt):
        glutPostRedisplay()
        if self._callback_timer_function:
            self._callback_timer_function(pt)

    def gl_run(self):
        assert self._width > 0
        glutMainLoop()
