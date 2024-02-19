export CARLA_ROOT="/home/enrico/Projects/Carla/CARLA_0.9.15"
export SCENARIO_RUNNER_ROOT="/home/enrico/Projects/Carla/LEADERBOARD_STUFF/scenario_runner"
export LEADERBOARD_ROOT="/home/enrico/Projects/Carla/LEADERBOARD_STUFF/leaderboard"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"/home/enrico/Projects/Carla/CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg":${PYTHONPATH}

# python try_to_show_log.py
# python try_normal_carla.py
# python try_to_record_something.py

export NUMBER_OF_VEHICLE=10

python setup_world.py
python /home/enrico/Projects/Carla/CARLA_0.9.15/PythonAPI/examples/generate_traffic.py --number-of-vehicles 300 --safe --hybrid --car-lights-on --respawn --hero