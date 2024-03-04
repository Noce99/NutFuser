from termcolor import colored

def color_error_string(string):
    return colored(string, "red", attrs=["bold"]) # , "blink"

def color_info_string(string):
    return colored(string, "yellow", attrs=["bold"])
