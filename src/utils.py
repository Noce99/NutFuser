from termcolor import colored

def color_error_string(string):
    return colored(string, "red", attrs=["bold"]) # , "blink"
