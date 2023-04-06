import cv2
from facenet_pytorch import MTCNN


def main():
    # create MTCNN object with default settings
    detector = MTCNN()

    # create video capture object
    path_test_video = "/home/dhgbao/Research_Monash/dataset/ccu-data/CCU_COMBINED_ANNOTATED_VIDEO/all_videos/M01004JM6_0005.mp4"

    cap = cv2.VideoCapture(path_test_video)

    # get the video writer object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        # read a frame from the video
        ret, frame = cap.read()

        if ret:
            # detect faces in the frame
            faces = detector.detect(frame)

            # draw bounding boxes on the detected faces
            for face in faces[0]:
                x1, y1, x2, y2 = map(int, face)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # write the frame with bounding boxes to the output video
            out.write(frame)

            # # display the frame with bounding boxes
            # cv2.imshow('frame', frame)

            # # exit if the user presses the 'q' key
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break

    # release the video capture and writer objects, and close all windows
    cap.release()
    out.release()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
