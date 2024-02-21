# python try_to_show_log.py
# python try_normal_carla.py
# python try_to_record_something.py

export NUMBER_OF_VEHICLE=10

python setup_world.py
python /home/enrico/Projects/Carla/CARLA_0.9.15/PythonAPI/examples/generate_traffic.py --number-of-vehicles ${NUMBER_OF_VEHICLE} --safe --hybrid --car-lights-on --respawn --hero
