import os
import sys

def add_carla_library_to_path(carla_path, end_of_egg_file):
    """
    Check Integrity of the Carla Path
    """
    # (1) Check that the Carla's Path really exists
    if not os.path.isdir(carla_path):
        raise Exception(f"The given Carla Path doesn't exist! [{carla_path}]")
    # (2) Check that the egg file is really present and it works: being able to import carla!
    carla_pythonapi_dist_path = os.path.join(carla_path, "PythonAPI/carla/dist")
    if not os.path.isdir(carla_pythonapi_dist_path):
        raise Exception(f"The given Carla doen't contains a PythonAPI! [{carla_pythonapi_dist_path}]")
    egg_files = [file for file in os.listdir(carla_pythonapi_dist_path) if file[-len(end_of_egg_file):] == end_of_egg_file]
    if len(egg_files) == 0:
        raise Exception(f"The given Carla doen't contains a \"*{end_of_egg_file}\" file in \"{carla_pythonapi_dist_path}\"")
    if len(egg_files) > 1:
        raise Exception(f"The given Carla contains to many \"*{end_of_egg_file}\" files in \"{carla_pythonapi_dist_path}\"\n" +
                                                  "Set a more restrict search with the \"--end_of_egg_file\" arguments!")
    egg_file_path = os.path.join(carla_pythonapi_dist_path, egg_files[0])
    # Now that we have a unique egg file we add it to the python path!
    sys.path.append(egg_file_path)
    # (3) Check that the CarlaUE4 executable is present
    carlaUE4_folder = os.path.join(carla_path, "CarlaUE4/Binaries/Linux/")
    if not os.path.isdir(carlaUE4_folder):
        raise Exception(f"The folder in wicth I was expecting \"CarlaUE4-Linux-Shipping\" doesn't exists! [{carlaUE4_folder}]")
    files = os.listdir(carlaUE4_folder)
    if "CarlaUE4-Linux-Shipping" not in files:
        raise Exception(f"I cannot find \"CarlaUE4-Linux-Shipping\" executable in \"{carlaUE4_folder}\"!")
    carlaUE4_path = os.path.join(carlaUE4_folder, "CarlaUE4-Linux-Shipping")
    return egg_file_path, carlaUE4_path