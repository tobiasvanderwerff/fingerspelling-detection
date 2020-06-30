import xml.etree.ElementTree as ET
from typing import List, Tuple


def ms_to_frame_no(ms, fps):
    return int(ms // (1000 / fps))


def timeslot_to_ffmpeg_format(timeslot):
    total_secs = int(timeslot * 10**(-3))
    secs = total_secs % 60
    mins = (total_secs - secs) // 60
    return str(mins) + ":" + str(secs) + "." + str(timeslot % 1000)


class EAFParser:
    def __init__(self, eaf_path, signer):
        self.eaf_path = eaf_path
        self.signer = signer
        self.root = None
        self.glossR = None
        self.segment_count = None
        self.timeslots = None
        self.annotations = []

    def parse_fingerspelling_timeslots(self, min_samples, fps):
        """ Parse a EAF file for fingerspelling occurrences in the corresponding video.
            Output: list of tuples (begin, end), containing begin and end time of fingerspelling sequences in ms. """
        self.root = ET.parse(self.eaf_path).getroot()
        for tier in self.root.findall("TIER"):
            if tier.get("PARTICIPANT") == self.signer:
                if "GlossR" in tier.get("TIER_ID"):
                    self.glossR = tier
                    break
        if self.glossR is not None:
            timeslots = self.__parse_glossR(min_samples, fps)
            self.segment_count = len(timeslots)
            return timeslots

    def __time_slot_val(self, slot_no):
        return int(self.root.find("TIME_ORDER")[slot_no - 1].get("TIME_VALUE"))

    def __parse_glossR(self, min_samples, fps) -> List[Tuple]:
        self.timeslots = []
        for annotation in self.glossR:
            try:
                if annotation[0][0].text[0] == "#":
                    at = annotation[0][0].text
                    self.annotations.append(at)
                    ts1 = int(annotation[0].get("TIME_SLOT_REF1")[2:])
                    ts2 = int(annotation[0].get("TIME_SLOT_REF2")[2:])
                    ts1_val = self.__time_slot_val(ts1)
                    ts2_val = self.__time_slot_val(ts2)
                    if (ts2_val - ts1_val) > (1000 * (min_samples - 1) / fps):
                        self.timeslots.append((ts1_val, ts2_val))
            except TypeError:
                continue
        return self.timeslots
