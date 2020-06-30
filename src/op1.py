import sys

""" Wrapper class for OpenPose API. """


try:
    sys.path.append('../../openpose/build/python');
    from openpose import pyopenpose as op
except Exception as e:
    print(e)
    sys.exit(-1)


class OpenPose:
    def __init__(self, params):
        self.opWrapper = op.WrapperPython()
        self.params = params

    def start(self):
        self.opWrapper.configure(self.params)
        self.opWrapper.start()

    def process(self, img):
        datum = op.Datum()
        datum.cvInputData = img
        self.opWrapper.emplaceAndPop([datum])
        return datum
