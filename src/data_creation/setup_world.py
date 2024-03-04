import sys
sys.path.append("/leonardo_work/IscrC_SSNeRF/CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")
import carla 
import random
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
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
            print(f"The argument was an integer! [{town_int}] -> {config.TOWN_DICT[town_int]}")
    except Exception as e:
        print(e)
        print(f"The argument was not an integer [{sys.argv[1]}], I will setup Town15!")
        town_int = 15
    if len(sys.argv) == 2:
        print("No port given! I will set it to 20000!")
        carla_port = 20000
    else:
        carla_port = int(sys.argv[2])

client = carla.Client('localhost', carla_port)
client.set_timeout(240.0)
world = client.load_world(config.TOWN_DICT[town_int])
config.SELECTED_TOWN_NAME = config.TOWN_DICT[town_int]
spawn_point = random.choice(world.get_map().get_spawn_points()) 
world.get_spectator().set_transform(spawn_point)
