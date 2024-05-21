import sys
import os
from tqdm import tqdm
from datetime import datetime

def start_recording(carla_egg_path, recorders_folder_path, town, frames_to_record, rpc_port, you_can_tick, finished_recording_event):
    sys.path.append(carla_egg_path)
    try:
        import carla
    except:
        pass
    now = datetime.now()
    current_time = now.strftime("%d_%m_%Y_%H:%M:%S")
    file_name = f"{town}_{current_time}_{frames_to_record}.record"
    recorder_file_path = os.path.join(recorders_folder_path, file_name)

    # Connect the client and set up bp library
    client = carla.Client('localhost', rpc_port)
    client.set_timeout(60.0)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1. / 20
    # In this case, the simulator will take 20 steps (1/0.05) to recreate one second of
    # the simulated world.
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 10
    # fixed_delta_seconds <= max_substep_delta_time * max_substeps
    # In order to have an optimal physical simulation,
    # the substep delta time should at least be below 0.01666 and ideally below 0.01.
    world.apply_settings(settings)

    client.start_recorder(recorder_file_path)

    you_can_tick.set()
    for _ in tqdm(range(frames_to_record)):
        world.wait_for_tick()
        you_can_tick.set()

    client.stop_recorder()
    finished_recording_event.set()

def play_record(carla_egg_path, record_path, rpc_port, finished_playing_event):
    sys.path.append(carla_egg_path)
    try:
        import carla
    except:
        pass

    # Connect the client
    client = carla.Client('localhost', rpc_port)
    client.set_timeout(60.0)

    print("Reading the record file...")
    recorder_str = client.show_recorder_file_info(record_path, False)

    # TMP
    all_lines = recorder_str.split("\n")
    for i in range(len(all_lines)):
        if "hero" in all_lines[i]:
            for l in range(i-50, i+1):
                print(all_lines[l])
    # END TMP


    start_of_the_file = recorder_str[:100]
    end_of_the_file = recorder_str[-100:]
    start_lines = start_of_the_file.split("\n")
    end_lines = end_of_the_file.split("\n")
    recorder_map = start_lines[1][5:]
    frames = int(end_lines[-3][8:])
    print(f"Map: {recorder_map}")
    print(f"Frames: {frames}")

    client.load_world(recorder_map)

    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1. / 20
    # In this case, the simulator will take 20 steps (1/0.05) to recreate one second of
    # the simulated world.
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 10
    # fixed_delta_seconds <= max_substep_delta_time * max_substeps
    # In order to have an optimal physical simulation,
    # the substep delta time should at least be below 0.01666 and ideally below 0.01.
    world.apply_settings(settings)
    

    client.replay_file( record_path,
                        0, # START
                        0, # DURATION
                        47) # FOLLOW_ID

    world.tick()
    hero = None

    print("Waiting for the ego vehicle...")
    possible_vehicles = world.get_actors().filter('vehicle.*')
    for vehicle in possible_vehicles:
        if vehicle.attributes['role_name'] == 'hero':
            print("Ego vehicle found")
            hero = vehicle
            break
    print(f"HERO: {hero.id}")
    
    for _ in tqdm(range(frames)):
        world.tick()

    finished_playing_event.set()