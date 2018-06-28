#!/usr/bin/env python

import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import PIL.Image
import numpy
import numpy as np
#import operator

#db = "haarcascade_frontalface_default.xml"
#db = "haarcascade_frontalface_alt.xml"
#db = "haarcascade_frontalcatface_extended.xml"
#db = "haarcascade_fullbody.xml"
db = "haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(db)

bridge = CvBridge()

face_marker = cv2.imread('smiley-1635449_960_720.png', cv2.IMREAD_UNCHANGED)

face_marker[:,:,3] -= face_marker[:,:,3] / 3

depth = False

color_image = False

cv2.namedWindow("image", cv2.WINDOW_NORMAL)

def image_callback(data):
    global bridge, face_cascade, depth, color_image

    try:
        image = bridge.imgmsg_to_cv2(data, "bgra8")
    except CvBridgeError as e:
        print(e)
        return

    color_image = image

def floodFill2(image, mask, point, value):
    if image[point[1], point[0]] > 0:
        cv2.floodFill(image, mask, point, value)

last_faces = False

def update():
    #print "update"

    global bridge, face_cascade, depth, color_image, last_faces

    image = color_image
    if image is False:
        print "no image"
        return
    image = image.copy()

    image = cv2.resize(image, (0,0), fx=1.1, fy=1.1)
    y = (image.shape[1] - color_image.shape[1]) / 2
    x = (image.shape[0] - color_image.shape[0]) / 2
    w = color_image.shape[1]
    h = color_image.shape[0]
    image = image[y:y+h,x:x+w,:]
    print x, y, w, h, image.shape

    #a = cv2.getRotationMatrix2D((m.shape[0]/2, m.shape[1]/2), rospy.Time.now().to_sec() * 300, 1)
    #image = cv2.warpAffine(image, a, (image.shape[0], image.shape[1]))


    if depth is False:
        print "no depth"
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow("gray", gray)
    cv2.waitKey(1)

    faces = face_cascade.detectMultiScale(gray, 1.1, 2)

    if last_faces is not False and len(last_faces) > 0:

      #faces = [np.argmin(np.array(last_faces)[:,0:2] - np.array(face)[0:2]) for face in faces]

      for face in faces:
        best_match = last_faces[0]
        for face_prev in last_faces:
          if best_match is not False and np.linalg.norm(np.array(best_match)[0:2] - np.array(face)[0:2]) > np.linalg.norm(np.array(face_prev)[0:2] - np.array(face)[0:2]):
            best_match = face_prev
        if best_match is not False:
          face[2] = face[2] * 0.2 + best_match[2] * 0.8
          face[3] = face[3] * 0.2 + best_match[3] * 0.8

    last_faces = faces

    d = depth.copy()

    #d[d > 0.0] = 0.0
    d[:,:] = 1024
    m = depth > 0.0
    d[m] = depth[m]

    d = cv2.medianBlur(d, 7)
    d = cv2.erode(d, np.ones((3,3),np.uint8), iterations = 15)

    d[d == 1024] = 0

    dx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize = 5)
    dy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize = 5)
    edges = cv2.max(cv2.max(dx, -dx), cv2.max(dy, -dy))
    edges = edges * 0.1

    #d[edges > 0.5] = 0.0

    cv2.imshow("edges", edges)

    #edges = cv2.dilate(edges, np.ones((3,3),np.uint8), iterations = 7

    #d2 = d.copy()
    d2 = np.zeros(d.shape[:2], np.uint8)
    d2[:,:] = 255
    d2[edges > 1.0] = 0
    d2[d <= 0.0] = 0

    mask = np.zeros((d.shape[0] + 2, d.shape[1] + 2), np.uint8)


    w = d2.shape[1] - 1
    h = d2.shape[0] - 1

    if True:
        t = 10
        for f in np.arange(0, 1, 0.01):
            floodFill2(d2, mask, (int((w - t - t) * f + t), t), 0)
            #floodFill2(d2, mask, (int((w - d - d) * f + d), h - d), 0)
            floodFill2(d2, mask, (t, int((h - t - t) * f + t)), 0)
            floodFill2(d2, mask, (w - t, int((h - t - t) * f + t)), 0)

    #d2 = cv2.medianBlur(d2, 7)
    #d2 = cv2.dilate(d2, np.ones((3, 3),np.uint8), iterations = 7)

    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

    faces2 = [ ]

    (clustercount, clusters, stats, centroids) = cv2.connectedComponentsWithStats(d2, 8, cv2.CV_32S)
    clusters2 = np.zeros(clusters.shape, np.uint8)
    clusters2[:,:] = clusters[:,:] * 100
    cv2.imshow("clusters", clusters2)

    for (x,y,w,h) in faces:
      if d2[y, x + w / 2] > 0 and d2[y, x + w / 2] != 100:
        #cv2.floodFill(d2, mask, (x + w / 2, y + h / 2), 100, 50)
        cv2.floodFill(d2, mask, (x + w / 2, y), 100, 50)
        faces2.append((x, y, w, h))



    cv2.imshow("depth", d * 100.0)



    cv2.imshow("depth2", d2)

    d2 = cv2.erode(d2, np.ones((3,3),np.uint8), iterations = 15)

    image2 = image.copy()
    image2[d2 == 100,0] = 0
    image2[d2 == 100,1] = 255
    image2[d2 == 100,2] = 0

    image = image / 2 + image2 / 2


    image_d = d2
    image_d = cv2.cvtColor(image_d, cv2.COLOR_GRAY2BGR)

    image_c = image.copy()
    for (x,y,w,h) in faces:
      #cv2.ellipse(image_c, (x, y), (w, h), 0, 0, 0, (0,0,255), 4)
      s = w / 2 + h / 2
      cv2.circle(image_c, (x + w / 2, y + h / 2), s / 2, (0,0,255), 4)
      cv2.circle(image_d, (x + w / 2, y + h / 2), s / 2, (0,0,255), 4)

    #image_c[edges > 0.1,:] = 0;

    for (x,y,w,h) in faces2:

        #if d2[y, x] > 0:
        #cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)
        #m = cv2.resize(face_marker, (image.shape[1], image.shape[0]))

        #x -= w / 4
        #y -= h / 4
        #y -= h / 6
        #w += w / 2
        #h += h / 2

        x -= w / 2
        y -= h / 2
        y -= h / 5
        w += w / 1
        h += h / 1

        if x < 0 or y < 0 or x + w >= image.shape[1] or y + h >= image.shape[0]:
            continue

        m = cv2.resize(face_marker, (w, h))

        a = cv2.getRotationMatrix2D((m.shape[0]/2, m.shape[1]/2), rospy.Time.now().to_sec() * 100, 1)
        m = cv2.warpAffine(m, a, (m.shape[0], m.shape[1]))

        m = cv2.copyMakeBorder(m, y, image.shape[0] - y - h, x, image.shape[1] - x - w, cv2.BORDER_CONSTANT, (0,0,0,0))
        #print m.shape, image.shape
        image = numpy.asarray(PIL.Image.alpha_composite(PIL.Image.fromarray(image), PIL.Image.fromarray(m)))


    image_c = image_c * (1.0 / 255)
    image_c[:,:,1] = depth

    cv2.imshow("image_c", image_c)

    cv2.imshow("image_d", image_d)

    cv2.imshow("depth", depth)

    cv2.imshow("image", image)

    cv2.waitKey(1)



image_sub = rospy.Subscriber("/camera/rgb/image_rect_color", Image, image_callback)


def depth_callback(data):
    global bridge, depth

    try:
        image = bridge.imgmsg_to_cv2(data)
    except CvBridgeError as e:
        print(e)
        return

    depth = image

    #cv2.imshow("depth", image)
    #cv2.waitKey(1)

depth_sub = rospy.Subscriber("/camera/depth_registered/image_raw", Image, depth_callback)


rospy.init_node("faces", anonymous=True)
#rospy.spin()

while not rospy.is_shutdown():
  #rospy.spinOnce()
  update()
