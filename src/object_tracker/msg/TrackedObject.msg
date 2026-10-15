# Single tracked object (matches takt's ByteTrack)

int32 track_id
float32 score
int32 class_id
int32 age_frames  # frames since the track was last associated

# Axis-aligned box in source image pixels (x1,y1,x2,y2)
float32 x1
float32 y1
float32 x2
float32 y2