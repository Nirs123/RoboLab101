from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

LEADER_PORT="/dev/ttyACM0"
FOLLOWER_PORT="/dev/ttyACM1"
WRIST_PATH="/dev/video4"
OVERHEAD_PATH="/dev/video2"

FOLLOWER_NAME="Follower_01_DarkGreen"
LEADER_NAME="Leader_01_Purple"

CAMERA_WIDTH=640
CAMERA_HEIGHT=480
CAMERA_FPS=30

camera_config = {
    "wrist": OpenCVCameraConfig(
        index_or_path=WRIST_PATH,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS
    ),
    "overhead": OpenCVCameraConfig(
        index_or_path=OVERHEAD_PATH,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS
    )
}

robot_config = SO101FollowerConfig(
    port=FOLLOWER_PORT,
    id=FOLLOWER_NAME,
    cameras=camera_config
)

teleop_config = SO101LeaderConfig(
    port=LEADER_PORT,
    id=LEADER_NAME
)

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)

robot.connect()
teleop_device.connect()

while True:
    observation = robot.get_observation()
    action = teleop_device.get_action()
    robot.send_action(action)
