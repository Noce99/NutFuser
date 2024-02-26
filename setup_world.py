import sys
sys.path.append("/leonardo_work/IscrC_SSNeRF/CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")
import carla 
import random

import config

if len(sys.argv) == 1:
    print("No argumnets I will setup Town15!")
    town_int = 15
else:
    try:
        town_int = int(sys.argv[1])
        if town_int not in config.TOWN_DICT.keys():
            print(f"This town does not exist [{town_int}], I will setup Town15!")
            town_int = 15
        else:
            print(f"The argument was an integer! [{town_int}] -> {town_dict[town_int]}")
    except:
        print("The argument was not an integer, I will setup Town15!")
        town_int = 15

client = carla.Client('localhost', 2000)
client.set_timeout(240.0)
world = client.load_world(config.TOWN_DICT[town_int])
config.SELECTED_TOWN_NAME = config.TOWN_DICT[town_int]
spawn_point = random.choice(world.get_map().get_spawn_points()) 
world.get_spectator().set_transform(spawn_point)
