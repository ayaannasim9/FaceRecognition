from collections import deque
from dataclasses import field,dataclass
@dataclass
class FaceTrack:
    track_id:int
    box:tuple[int,int,int,int]
    name:str="Unknown"
    prediction_history: deque = field(
        default_factory=lambda: deque(maxlen=3)
    )
    missed_frames:int=0

    def calculate_iou(self, box_B):
        xa, ya, wa, ha=self.box
        xb, yb, wb, hb=box_B

        intersection_left=max(xa, xb)
        intersection_top=max(ya, yb)
        intersection_right=min(xa+wa, xb+wb)
        intersection_bottom=min(ya+ha, yb+hb)

        intersection_width=max(0,intersection_right-intersection_left)
        intersection_height=max(0, intersection_bottom-intersection_top)

        intersection_area=(intersection_width*intersection_height)
        union_area=(wa*ha)+(wb*hb)-intersection_area

        if union_area==0:
            return 0.0

        iou=intersection_area/union_area
        return iou
