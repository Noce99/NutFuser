import carla 
import random

client = carla.Client('localhost', 2000)
client.set_timeout(240.0)
world = client.load_world("Town15")
spawn_point = random.choice(world.get_map().get_spawn_points()) 
world.get_spectator().set_transform(spawn_point)
