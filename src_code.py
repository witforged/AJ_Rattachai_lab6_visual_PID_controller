import cv2
import robomaster
from robomaster import robot
from robomaster import vision
from robomaster import blaster
import time
import pandas as pd
import matplotlib.pyplot as plt


class MarkerInfo:

    def __init__(self, x, y, w, h, info):
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        self._info = info

    @property
    def pt1(self):
        return int((self._x - self._w / 2) * 1280), int((self._y - self._h / 2) * 720)

    @property
    def pt2(self):
        return int((self._x + self._w / 2) * 1280), int((self._y + self._h / 2) * 720)

    @property
    def center(self):
        return int(self._x * 1280), int(self._y * 720)

    @property
    def text(self):
        return self._info


markers = []


# in case that there are many detected markers
def on_detect_marker(marker_info):
    number = len(marker_info)
    markers.clear()
    for i in range(0, number):
        x, y, w, h, info = marker_info[i]
        markers.append(
            MarkerInfo(x, y, w, h, info)
        )  # x and y w h is in a range of [0 1] relative to image's size
        # print("marker:{0} x:{1}, y:{2}, w:{3}, h:{4}".format(info, x, y, w, h))


def sub_data_handler(angle_info):
    global list_of_data
    list_of_data = angle_info
    # pitch_angle, yaw_angle, pitch_ground_angle, yaw_ground_angle = angle_info
    # print(
    #     "gimbal angle: pitch_angle:{0}, yaw_angle:{1}, pitch_ground_angle:{2}, yaw_ground_angle:{3}".format(
    #         pitch_angle, yaw_angle, pitch_ground_angle, yaw_ground_angle
    #     )
    # )


if __name__ == "__main__":
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_vision = ep_robot.vision
    ep_camera = ep_robot.camera
    ep_gimbal = ep_robot.gimbal
    ep_blaster = ep_robot.blaster
    # the image center constants
    center_x = 1280 / 2
    center_y = 720 / 2
    
    # initialize robot
    ep_camera.start_video_stream(display=False)
    ep_gimbal.sub_angle(freq=50, callback=sub_data_handler)
    result = ep_vision.sub_detect_info(
        name="marker", callback=on_detect_marker
    )  # sub_detect_info : new thread for camera detection
    # ep_gimbal.moveto(pitch=0, yaw=0).wait_for_completed()
    ep_gimbal.recenter(pitch_speed=200, yaw_speed=200).wait_for_completed()

    count = 0
    time.sleep(1)

    # PID controller constants
    p = -0.607  # -0.609 -0.65
    i = 0
    d = -0.00135  # -0.00135 0

    accumulate_err_x = 0
    accumulate_err_y = 0
    data_pith_yaw = []
    while True:
        if len(markers) != 0:  # target found
            after_time = time.time()
            x, y = markers[-1].center  # x,y here in the pixel unit

            err_x = (
                center_x - x
            )  # err_x = image_center in x direction - current marker center in x direction
            err_y = (
                center_y - y
            )  # err_y = image_center in y direction - current marker center in y direction
            # data_pith_yaw.append(list(list_of_data) + [err_x, err_y])
            accumulate_err_x += err_x
            accumulate_err_y += err_y

            # fire the laser pointer to the target
            # ep_blaster.fire(fire_type=blaster.INFRARED_FIRE)

            if count >= 1:
                speed_x = (
                    (p * err_x)
                    + d * ((prev_err_x - err_x) / (prev_time - after_time))
                    + i * (accumulate_err_x)
                )
                speed_y = (
                    (p * err_y)
                    + d * ((prev_err_y - err_y) / (prev_time - after_time))
                    + i * (accumulate_err_y)
                )
                ep_gimbal.drive_speed(pitch_speed=-speed_y, yaw_speed=speed_x)
                data_pith_yaw.append(
                    list(list_of_data)
                    + [err_x, err_y, round(speed_x, 3), round(speed_y, 3)]
                )

            count += 1
            prev_time = time.time()
            prev_err_x = err_x
            prev_err_y = err_y
            time.sleep(0.001)
        else:
            ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)

        img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        for j in range(0, len(markers)):
            cv2.rectangle(img, markers[j].pt1, markers[j].pt2, (0, 255, 0))
            cv2.putText(
                img,
                markers[j].text,
                markers[j].center,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3,
            )
        cv2.imshow("Markers", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

    result = ep_vision.unsub_detect_info(name="marker")
    cv2.destroyAllWindows()
    ep_camera.stop_video_stream()
    ep_robot.close()
    # data_pith_yaw = pd.DataFrame(data_pith_yaw)
    # data_pith_yaw.to_csv("data_pith_yaw.csv")

    # data_pith_yaw = data_pith_yaw
    x_point = [i for i in range(len(data_pith_yaw))]
    # y_point0 = [i[0] for i in data_pith_yaw]
    # plt.legend('pitch')
    # y_point1 = [i[1] for i in data_pith_yaw]
    # plt.legend('yaw')
    # y_point2 = [i[2] for i in data_pith_yaw]
    # plt.legend('pitch_g')
    # y_point3 = [i[3] for i in data_pith_yaw]
    # plt.legend('yaw_g')
    y_point4 = [i[4] for i in data_pith_yaw]
    # plt.legend("error x")
    y_point5 = [i[5] for i in data_pith_yaw]
    # plt.legend("error y")
    y_point6 = [i[6] for i in data_pith_yaw]
    # plt.legend("u x")
    y_point7 = [i[7] for i in data_pith_yaw]

    # plt.plot(x_point, y_point0)
    # plt.plot(x_point, y_point1)
    # plt.plot(x_point, y_point2)
    # plt.plot(x_point, y_point3)
    plt.plot(x_point, y_point4)
    plt.plot(x_point, y_point5)
    plt.plot(x_point, y_point6)
    plt.plot(x_point, y_point7)
    plt.legend(["e x", "e y", "u x", "u y"])
    plt.show()




อธิบายโค้ดนี้เเบบละเอียด