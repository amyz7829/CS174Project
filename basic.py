import sys
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image
import numpy as np
import cv2
import math

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15
class Test:

    def __init__(self):
        self.view_matrix = None

        self.orig_img = None
        self.ref_img = None

        self.texture_background = None

    def _init_gl(self, Width, Height):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(Width) / float(Height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

        # enable texture
        glEnable(GL_TEXTURE_2D)
        self.texture_background = glGenTextures(1)
        self.texture_cube = glGenTextures(1)

        # create cube texture
        image = Image.open("red.jpg")
        ix = image.size[0]
        iy = image.size[1]
        image = image.tobytes("raw", "RGBX", 0, -1)

        glBindTexture(GL_TEXTURE_2D, self.texture_cube)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)

    def _draw_cube(self):
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0);
        glVertex3f(-1.0, -1.0, 1.0)
        glTexCoord2f(1.0, 0.0);
        glVertex3f(1.0, -1.0, 1.0)
        glTexCoord2f(1.0, 1.0);
        glVertex3f(1.0, 1.0, 1.0)
        glTexCoord2f(0.0, 1.0);
        glVertex3f(-1.0, 1.0, 1.0)
        glTexCoord2f(1.0, 0.0);
        glVertex3f(-1.0, -1.0, -1.0)
        glTexCoord2f(1.0, 1.0);
        glVertex3f(-1.0, 1.0, -1.0)
        glTexCoord2f(0.0, 1.0);
        glVertex3f(1.0, 1.0, -1.0)
        glTexCoord2f(0.0, 0.0);
        glVertex3f(1.0, -1.0, -1.0)
        glTexCoord2f(0.0, 1.0);
        glVertex3f(-1.0, 1.0, -1.0)
        glTexCoord2f(0.0, 0.0);
        glVertex3f(-1.0, 1.0, 1.0)
        glTexCoord2f(1.0, 0.0);
        glVertex3f(1.0, 1.0, 1.0)
        glTexCoord2f(1.0, 1.0);
        glVertex3f(1.0, 1.0, -1.0)
        glTexCoord2f(1.0, 1.0);
        glVertex3f(-1.0, -1.0, -1.0)
        glTexCoord2f(0.0, 1.0);
        glVertex3f(1.0, -1.0, -1.0)
        glTexCoord2f(0.0, 0.0);
        glVertex3f(1.0, -1.0, 1.0)
        glTexCoord2f(1.0, 0.0);
        glVertex3f(-1.0, -1.0, 1.0)
        glTexCoord2f(1.0, 0.0);
        glVertex3f(1.0, -1.0, -1.0)
        glTexCoord2f(1.0, 1.0);
        glVertex3f(1.0, 1.0, -1.0)
        glTexCoord2f(0.0, 1.0);
        glVertex3f(1.0, 1.0, 1.0)
        glTexCoord2f(0.0, 0.0);
        glVertex3f(1.0, -1.0, 1.0)
        glTexCoord2f(0.0, 0.0);
        glVertex3f(-1.0, -1.0, -1.0)
        glTexCoord2f(1.0, 0.0);
        glVertex3f(-1.0, -1.0, 1.0)
        glTexCoord2f(1.0, 1.0);
        glVertex3f(-1.0, 1.0, 1.0)
        glTexCoord2f(0.0, 1.0);
        glVertex3f(-1.0, 1.0, -1.0)
        glEnd()

    def _draw_scene(self):
        height, width, channels = self.orig_img.shape
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glPushMatrix()
        glTranslatef(0, 0, -11)
        self._draw_background()
        glPopMatrix()

        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glEnable(GL_BLEND)
        glDisable(GL_DEPTH_TEST)

        glBindTexture(GL_TEXTURE_2D, self.texture_cube)
        glPushMatrix()
        glTranslatef(0, 0, -11)
        glRotatef(20, 0, 1, 0)
        self._draw_cube()
        glPopMatrix()

        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)

        glutSwapBuffers()

    def _setup_background(self, filepath):
        # convert image to OpenGL texture format
        image = Image.open(filepath)
        image = image.rotate(180)
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        ix = image.size[0]
        iy = image.size[1]
        if ".png" in filepath:
            image = image.tobytes("raw", "RGBA", 0, -1)
        else:
            image = image.tobytes("raw", "RGBX", 0, -1)

        # create background texture
        glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)

    def _draw_background(self):
        height, width, channels = self.orig_img.shape

        bg_height_factor = height / width
        # draw background
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0);
        glVertex3f(-1, -1 * bg_height_factor, 0)
        glTexCoord2f(1.0, 1.0);
        glVertex3f(1, -1 * bg_height_factor, 0)
        glTexCoord2f(1.0, 0.0);
        glVertex3f(1, bg_height_factor, 0)
        glTexCoord2f(0.0, 0.0);
        glVertex3f(-1, bg_height_factor, 0)
        glEnd()

    def main(self):
        # setup and run OpenGL
        if len(sys.argv) == 2:
            self.orig_img = cv2.imread(sys.argv[1])
            height, width, channels = self.orig_img.shape

            glutInit()
            glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
            glutInitWindowSize(width, height)
            glutInitWindowPosition(800, 400)
            glutCreateWindow("OpenGL Test")
            glutDisplayFunc(self._draw_scene)
            glutIdleFunc(self._draw_scene)
            self._init_gl(width, height)
            self._setup_background(sys.argv[1])
            glutMainLoop()
        else:
            print("Usage: <img> <ref img>")



# run instance of Hand Tracker
test = Test()
test.main()