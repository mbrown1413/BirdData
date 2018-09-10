
import os
import multiprocessing
from time import time, sleep
from collections import namedtuple
from functools import partial
from itertools import count

import cv2
import numpy

VIDEO_SOURCE = 0  # Index of v4l camera

# Settings for DHT temp/humidity sensor
# See AdafruitDHT library for details on sensor and pin settings.
DHT_ENABLE = False
if DHT_ENABLE:
    import Adafruit_DHT
    DHT_SENSOR = Adafruit_DHT.DHT22  # DHT sensor model
    DHT_PIN = '3'  # Raspberry pi I/O port pin of DHT sensor

def get_out_filename():
    dirname = 'data'
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    for i in count():
        filename = os.path.join(dirname, 'bird_{}.csv'.format(i))
        if not os.path.exists(filename):
            return filename

class AsyncDhtReader:

    def __init__(self, sensor_type, pin, delay=0):
        self.sensor_type = sensor_type
        self.pin = pin
        self.delay = delay

        self.shared = multiprocessing.Manager().Namespace()
        self.shared.humid = None
        self.shared.temp = None

        self.process = multiprocessing.Process(target=self._process)
        self.process.start()

    def _process(self):
        while True:
            humid, temp = Adafruit_DHT.read_retry(DHT_SENSOR, DHT_PIN)
            if None not in (temp, humid):
                self.shared.humid, self.shared.temp = humid, temp
                if self.delay:
                    sleep(self.delay)

    def read(self):
        return self.shared.humid, self.shared.temp

    def close(self):
        if getattr(self, 'process', None):
            self.process.terminate()

    __del__ = close

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
        if n_components <= 1:
            return None, mask
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
        raise RuntimeError('Could not open video source "{}"'.format(VIDEO_SOURCE))
    if debug:
        cv2.namedWindow('frame')
    hat_finder = HatFinder(
        # Adjustable color parameters
        h_min=55, h_max=180,
        s_min=0, s_max=158,
        v_min=25, v_max=175,
        debug_window='debug' if debug else None,
    )

    if DHT_ENABLE:
        dht_reader = AsyncDhtReader(DHT_SENSOR, DHT_PIN, 60)
    out_file = open(get_out_filename(), 'w')
    try:
        run(video, debug, out_file, hat_finder, dht_reader)
    finally:
        out_file.close()
        dht_reader.close()
        video.release()

def run(video, debug, out_file, hat_finder, dht_reader):

    # Setup data file
    start_time = time()
    out_file.write('# Start Time: {}\n'.format(start_time))
    out_file.write('# time,x,y,temperature,humidity\n')

    temp = humid = None
    prev_values = None
    while True:
        ret, frame = video.read()
        if not ret:
            print('Error reading from camera')
            break
        t = time()

        # Find bird hat
        centroid, mask = hat_finder.find(frame)
        if centroid is None:
            x, y = None, None
        else:
            x, y = tuple(map(int, map(round, centroid)))

        # Read temperature and humidity
        if dht_reader:
            humid, temp = dht_reader.read()

        # Format data values
        relative_time = t - start_time
        values = [
            '{0:0.1f}'.format(value) if value else ''
            for value in [relative_time, x, y, temp, humid]
        ]
        values = [v[:-2] if v.endswith('.0') else v for v in values]

        # Set values to blank if they haven't changed
        tmp_values = list(values)
        if prev_values:
            for i in range(len(values)):
                if values[i] == prev_values[i]:
                    values[i] = ''
        prev_values = tmp_values

        # Write data
        data_line = ','.join(values).rstrip(',')
        print(data_line)
        out_file.write(data_line+'\n')
        out_file.flush()

        if debug:

            if temp and humid:
                print('Temp={0:0.1f} Humid={1:0.1f}%'.format(temp, humid))
            else:
                print('Temp=? Humid=?'.format(temp, humid))

            if x and y:
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('debug', mask)
            cv2.imshow('frame', frame)

            key = cv2.waitKey(50)
            if key == 27:  # Escape
                break

if __name__ == '__main__':
    main()
