import math
import time
import cv2
import sys
import argparse
import pathlib
from nutfuser.carla_interface.run_carla import (check_integrity_of_carla_path,
                                                launch_carla_server_saifly_and_wait_till_its_up,
                                                launch_manual_control_saifly,
                                                set_up_traffic_manager_saifly_and_wait_till_its_up,
                                                set_up_world_saifly_and_wait_till_its_setted_up)
from nutfuser import utils, config
from ctypes import c_int
import os
import multiprocessing
import signal
import traceback

from carla_birdeye_view import (
    BirdViewProducer,
    BirdView,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    BirdViewCropType,
)
from carla_birdeye_view.mask import PixelDimensions

STUCK_SPEED_THRESHOLD_IN_KMH = 3
MAX_STUCK_FRAMES = 3000


def get_speed(actor) -> float:
    """in km/h"""
    vector = actor.get_velocity()
    MPS_TO_KMH = 3.6
    return MPS_TO_KMH * math.sqrt(vector.x ** 2 + vector.y ** 2 + vector.z ** 2)


def main(args, you_can_tick_event):
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    map = world.get_map()
    spawn_points = map.get_spawn_points()
    blueprints = world.get_blueprint_library()

    hero = None
    while hero is None:
        print("Waiting for the ego vehicle...")
        possible_vehicles = world.get_actors().filter('vehicle.*')
        for vehicle in possible_vehicles:
            if vehicle.attributes['role_name'] == 'hero':
                print("Ego vehicle found")
                hero = vehicle
                break
        time.sleep(0.1)
        if not args.no_traffic:
            you_can_tick_event.set()
            world.wait_for_tick(seconds=3.0)
    # hero.set_autopilot(True)

    birdview_producer = BirdViewProducer(
        client,
        PixelDimensions(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT),
        pixels_per_meter=16,
        crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
        render_lanes_on_junctions=True,
    )

    while True:
        if not args.no_traffic:
            you_can_tick_event.set()
            world.wait_for_tick()
        birdview: BirdView = birdview_producer.produce(agent_vehicle=hero)
        bgr = cv2.cvtColor(BirdViewProducer.as_rgb(birdview), cv2.COLOR_BGR2RGB)
        # NOTE imshow requires BGR color model
        cv2.imshow("BirdView RGB", bgr)

        # Play next frames without having to wait for the key
        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # ESC
            break
    cv2.destroyAllWindows()


pids_to_be_killed = []


def run_all(args):
    # (1) LAUNCH CARLA SERVER
    print("Launching Carla Server...")
    nutfuser_folder_path = pathlib.Path(__file__).parent.resolve()
    carla_log_path = os.path.join(nutfuser_folder_path, "logs", f"carla_server_logs_test_bird_eye_view.log")
    os.environ["PATH"] = f"{os.environ['PATH']}:/leonardo/home/userexternal/emannocc/xdg-user-dirs-0.18/"
    carla_server_pid = multiprocessing.Value(c_int)
    carla_was_correctly_started_up = launch_carla_server_saifly_and_wait_till_its_up(
        rpc_port=args.rpc_port,
        carla_server_pid=carla_server_pid,
        carlaUE4_path=carlaUE4_path,
        logs_path=carla_log_path,
        how_many_seconds_to_wait=10,
        show_carla_window=True
    )

    pids_to_be_killed.append(carla_server_pid.value)

    if not carla_was_correctly_started_up:
        raise utils.NutException(utils.color_error_string(f"Carla crashed while starting!"))

    print(utils.color_info_string("(1/4)\tCarla Server is UP!"))

    # (2) SET UP THE WORLD
    world_was_correctly_setted_up = set_up_world_saifly_and_wait_till_its_setted_up(
        carla_ip=args.carla_ip,
        rpc_port=args.rpc_port,
        town_number=args.town,
        carla_server_pid=carla_server_pid
    )

    if not world_was_correctly_setted_up:
        raise utils.NutException(utils.color_error_string(f"Failed to set up world!"))

    print(utils.color_info_string("(2/4)\tSet up the world!"))

    if not args.no_traffic:
        # (3) SET UP TRAFFIC MANAGER
        traffic_manager_log_path = os.path.join(nutfuser_folder_path, "logs", f"traffic_manager_logs_test_bird_eye_view.log")
        traffic_manager_pid = multiprocessing.Value(c_int)
        carla_is_ok, \
            traffic_manager_is_ok, \
            you_can_tick, \
            traffic_manager_is_up, \
            set_up_traffic_manager_process = set_up_traffic_manager_saifly_and_wait_till_its_up(
            carla_ip=args.carla_ip,
            rpc_port=args.rpc_port,
            tm_port=args.tm_port,
            number_of_vehicles=args.num_of_vehicle,
            number_of_walkers=args.num_of_walkers,
            carla_server_pid=carla_server_pid,
            traffic_manager_pid=traffic_manager_pid,
            logs_path=traffic_manager_log_path,
            hero=False
        )

        if not carla_is_ok:
            raise utils.NutException(utils.color_error_string(f"Carla crashed while setting up Traffic Manager!"))
        if not traffic_manager_is_ok:
            raise utils.NutException(utils.color_error_string(f"Traffic Manager Crashed!"))

        print(utils.color_info_string("(3/4)\tTraffic Manager Setted Up properly!"))

        pids_to_be_killed.append(traffic_manager_pid.value)
    else:
        you_can_tick = None

    # (4) LAUNCH MANUAL CONTROL
    print("Launching Manual Control...")
    manual_control_pid = multiprocessing.Value(c_int)
    carla_is_ok, \
        manual_control_is_ok, \
        manual_control_is_up = launch_manual_control_saifly(
        carla_server_pid=carla_server_pid,
        manual_control_pid=manual_control_pid
    )

    pids_to_be_killed.append(manual_control_pid.value)

    if not carla_is_ok:
        raise utils.NutException(utils.color_error_string(f"Carla crashed while setting up Manual Control!"))
    if not manual_control_is_ok:
        raise utils.NutException(utils.color_error_string(f"Manual Control Crashed!"))

    print(utils.color_info_string("(4/4)\tManual Control is UP!"))

    print(utils.color_info_string("Calculating and showing bird_eye_view"))

    main(args, you_can_tick)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--carla_ip',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)'
    )
    argparser.add_argument(
        '--rpc_port',
        default=2000,
        help='Carla RPC port (deafult: 2000)',
        type=int
    )
    argparser.add_argument(
        '--tm_port',
        default=8000,
        help='Traffic Manager port (default: 8000)',
        type=int
    )
    argparser.add_argument(
        '--end_of_egg_file',
        help='How the egg file should end to be valid! (default: py3.7-linux-x86_64.egg)',
        default="py3.7-linux-x86_64.egg",
        type=str
    )
    argparser.add_argument(
        '--carla_path',
        help='Path to the Carla Installation!',
        required=True,
        type=str
    )
    argparser.add_argument(
        '--num_of_vehicle',
        help='Number of Vehicle to spawn! (default: 30)',
        required=False,
        default=30,
        type=int
    )
    argparser.add_argument(
        '--num_of_walkers',
        help='Number of Walkers to spawn! (default: 30)',
        required=False,
        default=30,
        type=int
    )
    argparser.add_argument(
        '--town',
        default=10,
        help='Witch town to select (default: 10)',
        type=int
    )
    argparser.add_argument(
        '--no_traffic',
        help='Set if you don\'t want to run the Traffic Manager!',
        action='store_true'
    )
    args = argparser.parse_args()
    if args.town not in config.TOWN_DICT:
        error = f"Invalid Town Index! [{args.town}]\n" + \
                 "Possible Town Index:\n"
        for key in config.TOWN_DICT:
            error += f"{key} -> {config.TOWN_DICT[key]}\n"
        raise Exception(utils.color_error_string(error))
    egg_file_path, carlaUE4_path = check_integrity_of_carla_path(args)
    try:
        import carla
    except:
        raise Exception(utils.color_error_string(f"Not able to import Carla from [{egg_file_path}]"))

    try:
        run_all(args)
    except Exception as e:
        traceback.print_exc()
        for pid in pids_to_be_killed:
            try:
                os.kill(pid, signal.SIGKILL)
            except:
                pass
        pids_to_be_killed = []
        exit()

    for pid in pids_to_be_killed:
        try:
            os.kill(pid, signal.SIGKILL)
        except:
            pass
