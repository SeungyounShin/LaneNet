import cv2

vidcap = cv2.VideoCapture('/home/yo0n/바탕화면/kookmin_auto/1_2/xycar_simul/track-s.mkv' )
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  print(count)
  cv2.imwrite("/home/yo0n/바탕화면/LaneNet-master/kookmin/frame%d.jpg" % count, image)     # save frame as JPEG file

  count += 1
