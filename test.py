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

    # This function finds how we would transform the marker to the shape found in the given picture. It requires no
    # external arguments, and gives the matching points of the reference image and the given picture as well as their
    # homography
    def _find_homography(self):
        # Convert images to grayscale
        origGray = cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2GRAY)
        refGray = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2GRAY)

        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(refGray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(origGray, None)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # Draw top matches
        imMatches = cv2.drawMatches(self.ref_img, keypoints1,self.orig_img, keypoints2, matches, None)
        cv2.imwrite("matches.jpg", imMatches)

        # Extract location of good matches
        points_ref = np.zeros((len(matches), 2), dtype=np.float32)
        points_orig = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points_ref[i, :] = keypoints1[match.queryIdx].pt
            points_orig[i, :] = keypoints2[match.trainIdx].pt

        return points_orig, points_ref

    def _find_contours(self):
        # Convert images to grayscale
        gray = cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 100, 200)

        # Stage 3: Find contours
        im1, contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for c in contours:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # if our approximated contour has four points, then
            # we can assume that we have found our screen
            if len(approx) == 4:
                contour = approx
                final_contour = np.zeros((4, 2), np.float32)
                for i in range(0, 4):
                    final_contour[i] = contour[i][0]
                return final_contour

        return None

    def _find_view_matrix(self, approx):
        obj_points = np.zeros((4, 3), np.float32)

        obj_points[0][0] = -1
        obj_points[0][1] = 1

        obj_points[1][0] = 1
        obj_points[1][1] = 1

        obj_points[2][0] = 1
        obj_points[2][1] = -1

        obj_points[3][0] = -1
        obj_points[3][1] = -1

        gray = cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2GRAY)
        ret, cam_mat, dist, r, t = cv2.calibrateCamera(np.array([obj_points]), np.array([approx]),
                                                       gray.shape[::-1], None, None)
        # project 3D points to image plane
        ret, rvecs, tvecs = cv2.solvePnP(np.array([obj_points]), np.array([approx]), cam_mat, dist)
        tvecs[2] = tvecs[2]
        # build view matrix
        rmtx = cv2.Rodrigues(rvecs)[0]

        view_matrix = np.array([[rmtx[0][0], rmtx[0][1], rmtx[0][2], tvecs[0]],
                                [rmtx[1][0], rmtx[1][1], rmtx[1][2], tvecs[1]],
                                [rmtx[2][0], rmtx[2][1], rmtx[2][2], tvecs[2]],
                                [0.0, 0.0, 0.0, 1.0]])

        inverse_matrix = np.array([[1.0, 1.0, 1.0, 1.0],
                                   [-1.0, -1.0, -1.0, -1.0],
                                   [-1.0, -1.0, -1.0, -1.0],
                                   [1.0, 1.0, 1.0, 1.0]])

        view_matrix = view_matrix * inverse_matrix

        self.view_matrix = np.transpose(view_matrix)

    # This function finds the rotation / translation needed to get from the marker to the given image, and will use
    # these transformations on the cube
    def _find_rotation_translation(self, points_orig, points_ref):
        height, width, channels = self.ref_img.shape
        # Object points are the points on the original object. Given our original object should be flat, it shouldn't
        # have anything in the z-dimension. We want
        obj_points = np.zeros((len(points_ref), 3), np.float32)
        for i in range(len(points_ref)):
            obj_points[i][0] = points_ref[i][0] / width - 1
            obj_points[i][1] = points_ref[i][1] / height - 1

        gray = cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2GRAY)
        ret, cam_mat, dist, r, t = cv2.calibrateCamera(np.array([obj_points]), np.array([points_orig]), gray.shape[::-1], None, None)

        # project 3D points to image plane
        ret, rvecs, tvecs = cv2.solvePnP(np.array([obj_points]), np.array([points_orig]), cam_mat, dist)
        tvecs[2] = tvecs[2] + 11
        # build view matrix
        rmtx = cv2.Rodrigues(rvecs)[0]

        view_matrix = np.array([[rmtx[0][0], rmtx[0][1], rmtx[0][2], tvecs[0]],
                                [rmtx[1][0], rmtx[1][1], rmtx[1][2], tvecs[1]],
                                [rmtx[2][0], rmtx[2][1], rmtx[2][2], tvecs[2]],
                                [0.0, 0.0, 0.0, 1.0]])

        inverse_matrix = np.array([[1.0, 1.0, 1.0, 1.0],
                                   [-1.0, -1.0, -1.0, -1.0],
                                   [-1.0, -1.0, -1.0, -1.0],
                                   [1.0, 1.0, 1.0, 1.0]])

        view_matrix = view_matrix * inverse_matrix

        self.view_matrix = np.transpose(view_matrix)
        print(self.view_matrix)


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
        glTranslatef(0, 0, -width / 150)
        self._draw_background()
        glPopMatrix()

        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glEnable(GL_BLEND)
        glDisable(GL_DEPTH_TEST)

        glBindTexture(GL_TEXTURE_2D, self.texture_cube)
        glPushMatrix()
        glLoadMatrixd(self.view_matrix)
        glTranslatef(0, 0, 1)
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

        bg_width = width / 200
        bg_height = height / 200
        # draw background
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0);
        glVertex3f(-1 * bg_width, -1 * bg_height, 0)
        glTexCoord2f(1.0, 1.0);
        glVertex3f(bg_width, -1 * bg_height, 0)
        glTexCoord2f(1.0, 0.0);
        glVertex3f(bg_width, bg_height, 0)
        glTexCoord2f(0.0, 0.0);
        glVertex3f(-1 * bg_width, bg_height, 0)
        glEnd()

    def main(self):
        # setup and run OpenGL
        if len(sys.argv) == 3:
            self.orig_img = cv2.imread(sys.argv[1])
            height, width, channels = self.orig_img.shape
            self.ref_img = cv2.imread(sys.argv[2])

            self._find_view_matrix(self._find_contours())
            # points_orig, points_ref = self._find_homography()
            # self._find_rotation_translation(points_orig, points_ref)

            glutInit()
            glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
            glutInitWindowSize(600, int(600 * height / width))
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