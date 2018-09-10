
import os
from time import time
from collections import namedtuple
from functools import partial
from itertools import count

import cv2
import numpy

VIDEO_SOURCE = 1  # Index of v4l camera

# Settings for DHT temp/humidity sensor
# See AdafruitDHT library for details on sensor and pin settings.
DHT_ENABLE = False
if DHT_ENABLE:
    import Adafruit_DHT
    DHT_SENSOR = Adafruit_DHT.DHT22  # DHT sensor model
    DHT_PIN = '3'  # Raspberry pi I/O port pin of DHT sensor
    DHT_READ_INTERVAL = 30  # Seconds between reads of DHT sensor

def get_out_filename():
    dirname = 'data'
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    for i in count():
        filename = os.path.join(dirname, 'bird_{}.dat'.format(i))
        if not os.path.exists(filename):
            return filename

Param = namedtuple('Param', 'name default max')
class TrackbarAdjustable:
    """Allows adjustments to instance attributes by OpenCV."""
    parameters = ()

    def __init__(self, debug_window=None, **kwargs):

        self.debug_window = debug_window
        if self.debug_window:
            cv2.namedWindow(self.debug_window)

        for param in self.parameters:
            value = kwargs.pop(param.name, param.default)

            # Set default parameters
            if hasattr(self, param.name):
                raise ValueError('Param name clash: {}'.format(param.name))
            setattr(self, param.name, value)

            # Debug Trackbars
            if self.debug_window:
                cv2.createTrackbar(
                    param.name,
                    self.debug_window,
                    value,
                    param.max,
                    partial(setattr, self, param.name),
                )

        if kwargs:
            raise ValueError('Unknown kwargs: {}'.format(kwargs))

class HatFinder(TrackbarAdjustable):
    parameters = (
        Param('h_shift', 0, 1),
        Param('h_min', 0, 180), Param('h_max', 255, 180),
        Param('s_min', 0, 255), Param('s_max', 255, 255),
        Param('v_min', 0, 255), Param('v_max', 255, 255),
    )

    def find(self, img):

        # HSV Thresholds
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if self.h_shift:
            hsv = (hsv + 180) % 180
        hsv_min = numpy.array([self.h_min, self.s_min, self.v_min])
        hsv_max = numpy.array([self.h_max, self.s_max, self.v_max])
        mask = cv2.inRange(hsv, hsv_min, hsv_max)

        # Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find centroid by largest connected component
        n_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best_idx = None
        for i in range(1, n_components): # Ignore background label 0
            area = stats[i][cv2.CC_STAT_AREA]
            if best_idx is None or area >= stats[best_idx][cv2.CC_STAT_AREA]:
                best_idx = i

        return centroids[best_idx], mask

def main():
    debug = True

    # Camera setup
    video = cv2.VideoCapture(VIDEO_SOURCE)
    if not video.isOpened():
        raise RuntimeError('Could not open source "{}"'.format(VIDEO_SOURCE))
    cv2.namedWindow('frame')
    hat_finder = HatFinder(
        # Adjustable color parameters
        h_min=55, h_max=180,
        s_min=0, s_max=158,
        v_min=25, v_max=175,
        debug_window='debug' if debug else None,
    )

    # Setup data file
    start_time = time()
    out_file = open(get_out_filename(), 'w')
    out_file.write('# Start Time: {}'.format(start_time))
    out_file.write('# Time,Temperature,Humidity,Hat X,Hat Y')

    temp = humid = None
    last_dht_read = float('-inf')
    while True:
        ret, frame = video.read()
        if not ret:
            print('Error reading from camera')
            break
        t = time()

        # Find bird hat
        centroid, mask = hat_finder.find(frame)
        x, y = tuple(map(int, map(round, centroid)))

        # Read temperature and humidity
        if DHT_ENABLE and t + DHT_READ_INTERVAL >= last_dht_read:
            last_dht_read = t
            humid, temp = Adafruit_DHT.read_retry(DHT_SENSOR, DHT_PIN)

        # Write data file
        temp_str = temp or ''
        humid_str = humid or ''
        out_file.write(
            '{t},{temp_str},{humid_str},{x},{y}'.format(**locals())
        )
        out_file.flush()

        if debug:

            if temp and humid:
                print('Temp={0:0.1f} Humid={1:0.1f}%'.format(temp, humid))
            else:
                print('Temp=? Humid=?'.format(temp, humid))

            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('debug', mask)
            cv2.imshow('frame', frame)

            key = cv2.waitKey(50)
            if key == 27:  # Escape
                break

if __name__ == '__main__':
    main()
