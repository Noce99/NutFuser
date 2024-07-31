import argparse
import json
import os
import pathlib
from datetime import datetime
from tabulate import tabulate
# from nutfuser import utils

SELECTED_CRITERIA = ["RouteCompletionTest", "CollisionTest", "Duration"]


def put_jsons_in_couple(json_list, path_to_append):
    """
    Return a DICT like:
    {   '4': {
            'NO_flow':  'RouteScenario_42024-06-05-15-05-31.json',
            'flow':     'RouteScenario_42024-06-05-15-51-13.json'},
        '8': {
            'NO_flow':  'RouteScenario_82024-06-05-15-16-55.json',
            'flow':     'RouteScenario_82024-06-05-16-01-19.json'},
        '2': {
            'NO_flow':  'RouteScenario_22024-06-05-15-02-13.json',
            'flow':     'RouteScenario_22024-06-05-15-46-27.json'},
    }
    """
    json_in_couples = {}
    for a_json in json_list:
        # RouteScenario_92024-06-05-23-55-10.json
        # (1) LET'S GET THE SCENARIO NUMBER
        after_underscore = a_json.split("_")[1]  # 92024-06-05-23-55-10.json
        before_minus = after_underscore.split("-")[0]  # 92024
        scenario_num = before_minus[:-4]  # 9
        # (2) LET'S GET THE DATE
        before_dot = after_underscore.split(".")[0]  # 92024-06-05-23-55-10
        date_but_not_year = before_dot[len(before_dot.split("-")[0]):]
        year = before_dot.split("-")[0][-4:]
        full_date = year + date_but_not_year

        datetime_object = datetime.strptime(full_date, "%Y-%m-%d-%H-%M-%S")
        if scenario_num not in json_in_couples:
            json_in_couples[scenario_num] = (a_json, datetime_object)
        else:
            other_json_name, other_datetime_object = json_in_couples[scenario_num]
            if other_datetime_object < datetime_object:
                # other is No FLOW
                json_in_couples[scenario_num] = {"NO_flow": os.path.join(path_to_append, other_json_name),
                                                 "flow": os.path.join(path_to_append, a_json)}
            else:
                # other is FLOW
                json_in_couples[scenario_num] = {"NO_flow": os.path.join(path_to_append, a_json),
                                                 "flow": os.path.join(path_to_append, other_json_name)}
    return json_in_couples


def dict_of_json_paths(all_jsons, path_to_append):
    """
        Return a DICT like:
        {
            '4': 'RouteScenario_42024-06-05-15-05-31.json',
            '8': 'RouteScenario_82024-06-05-15-16-55.json',
            '2': 'RouteScenario_22024-06-05-15-46-27.json',
        }
        """
    parsed_json = {}
    for a_json in all_jsons:
        # RouteScenario_92024-06-05-23-55-10.json
        # (1) LET'S GET THE SCENARIO NUMBER
        after_underscore = a_json.split("_")[1]  # 92024-06-05-23-55-10.json
        before_minus = after_underscore.split("-")[0]  # 92024
        scenario_num = before_minus[:-4]  # 9
        parsed_json[scenario_num] = os.path.join(path_to_append, a_json)
    return parsed_json


def count_collisions_and_completion_percentage_couples(json_in_couples):
    total_collisions = {"NO_flow": 0, "flow": 0}
    total_completion_percentage = {"NO_flow": 0.0, "flow": 0.0}
    total_duration = {"NO_flow": 0.0, "flow": 0.0}
    for scenarios in json_in_couples:
        no_flow_json = json_in_couples[scenarios]["NO_flow"]
        flow_json = json_in_couples[scenarios]["flow"]
        parsed_no_flow = parse_json(no_flow_json)
        parsed_flow = parse_json(flow_json)
        assert parsed_no_flow["scenario_num"] == scenarios
        assert parsed_flow["scenario_num"] == scenarios
        for criteria in SELECTED_CRITERIA:
            assert parsed_no_flow["criteria"][criteria] is not None
            assert parsed_flow["criteria"][criteria] is not None
        total_collisions["NO_flow"] += parsed_no_flow["criteria"]["CollisionTest"]["actual"]
        total_collisions["flow"] += parsed_flow["criteria"]["CollisionTest"]["actual"]
        total_completion_percentage["NO_flow"] += parsed_no_flow["criteria"]["RouteCompletionTest"]["actual"]
        total_completion_percentage["flow"] += parsed_flow["criteria"]["RouteCompletionTest"]["actual"]
        total_duration["NO_flow"] += parsed_no_flow["criteria"]["Duration"]["actual"]
        total_duration["flow"] += parsed_flow["criteria"]["Duration"]["actual"]

        if abs(parsed_no_flow["criteria"]["CollisionTest"]["actual"] - parsed_flow["criteria"]["CollisionTest"][
            "actual"]) > 3:
            print(
                f'Interesting one! ({scenarios}) FLOW/NO_FLOW = {parsed_flow["criteria"]["CollisionTest"]["actual"]}/{parsed_no_flow["criteria"]["CollisionTest"]["actual"]}')

    average_collisions = {}
    for el in total_collisions:
        average_collisions[el] = total_collisions[el] / len(json_in_couples)
    average_completion_percentage = {}
    for el in total_completion_percentage:
        average_completion_percentage[el] = total_completion_percentage[el] / len(json_in_couples)
    average_duration = {}
    for el in total_duration:
        average_duration[el] = total_duration[el] / len(json_in_couples)

    total_simulation_time = total_duration["NO_flow"] + total_duration["flow"]

    return average_collisions, average_completion_percentage, average_duration, total_simulation_time


def count_collisions_and_completion_percentage(json_files):
    total_collisions = 0
    total_completion_percentage = 0
    total_duration = 0
    for scenarios in json_files:
        parsed_json = parse_json(json_files[scenarios])
        assert parsed_json["scenario_num"] == scenarios
        for criteria in SELECTED_CRITERIA:
            assert parsed_json["criteria"][criteria] is not None
        total_collisions += parsed_json["criteria"]["CollisionTest"]["actual"]
        total_completion_percentage += parsed_json["criteria"]["RouteCompletionTest"]["actual"]
        total_duration += parsed_json["criteria"]["Duration"]["actual"]

    average_collisions = total_collisions / len(json_files)
    average_completion_percentage = total_completion_percentage / len(json_files)
    average_duration = total_duration / len(json_files)

    return average_collisions, average_completion_percentage, average_duration, total_duration


def parse_json(a_json):
    result = {}
    with open(a_json, 'r') as f:
        data = json.load(f)
        scenario_num = data["scenario"][len("RouteScenario_"):]
        my_criteria = {element: None for element in SELECTED_CRITERIA}
        criteria = data["criteria"]
        for a_criteria in criteria:
            name = a_criteria["name"]
            expected = a_criteria["expected"]
            actual = a_criteria["actual"]
            if name in SELECTED_CRITERIA:
                my_criteria[name] = {"expected": expected, "actual": actual}
        result["scenario_num"] = scenario_num
        result["criteria"] = my_criteria
    return result


def print_results_couples(results):
    # (1) LET'S CREATE THE TABLE
    city_table_head = ["Town",
                       "Routes",
                       "Collisions No Flow",
                       "Collisions Flow",
                       "Completion Percentage No Flow",
                       "Completion Percentage Flow",
                       "Duration No Flow",
                       "Duration Flow"]
    city_table = []
    total_table_head = ["Collisions No Flow",
                        "Collisions Flow",
                        "Completion Percentage No Flow [%]",
                        "Completion Percentage Flow [%]",
                        "Duration No Flow [s]",
                        "Duration Flow [s]"]
    total_table = []
    tot_collisions_no_flow = 0
    tot_collisions_flow = 0
    tot_completion_percentage_no_flow = 0
    tot_completion_percentage_flow = 0
    tot_duration_no_flow = 0
    tot_duration_flow = 0
    for town in results:
        city_table.append([town.split("_")[0],
                           results[town]["num_of_routes"],
                           results[town]["collisions"]["NO_flow"],
                           results[town]["collisions"]["flow"],
                           results[town]["completion_percentage"]["NO_flow"],
                           results[town]["completion_percentage"]["flow"],
                           results[town]["duration"]["NO_flow"],
                           results[town]["duration"]["flow"], ])
        tot_collisions_no_flow += results[town]["collisions"]["NO_flow"]
        tot_collisions_flow += results[town]["collisions"]["flow"]
        tot_completion_percentage_no_flow += results[town]["completion_percentage"]["NO_flow"]
        tot_completion_percentage_flow += results[town]["completion_percentage"]["flow"]
        tot_duration_no_flow += results[town]["duration"]["NO_flow"]
        tot_duration_flow += results[town]["duration"]["flow"]
    total_table.append([tot_collisions_no_flow / len(results),
                        tot_collisions_flow / len(results),
                        tot_completion_percentage_no_flow / len(results),
                        tot_completion_percentage_flow / len(results),
                        tot_duration_no_flow / len(results),
                        tot_duration_flow / len(results)])
    # (2) LET'S PRINT ALL
    to_print = "# Singular City Results #"
    print(utils.color_info_string("#" * len(to_print)))
    print(utils.color_info_string(to_print))
    print(utils.color_info_string("#" * len(to_print)))
    print(tabulate(city_table, headers=city_table_head, tablefmt="grid"))
    to_print = "# Total Results #"
    print(utils.color_info_string("#" * len(to_print)))
    print(utils.color_info_string(to_print))
    print(utils.color_info_string("#" * len(to_print)))
    print(tabulate(total_table, headers=total_table_head, tablefmt="grid"))


def print_results(results):
    # (1) LET'S CREATE THE TABLE
    city_table_head = ["Town",
                       "Routes",
                       "Collisions",
                       "Completion Percentage",
                       "Duration"]
    city_table = []
    total_table_head = ["Collisions",
                        "Completion Percentage [%]",
                        "Duration [s]"]
    total_table = []
    tot_collisions = 0
    tot_completion_percentage = 0
    tot_duration = 0
    for town in results:
        city_table.append([town.split("_")[0],
                           results[town]["num_of_routes"],
                           results[town]["collisions"],
                           results[town]["completion_percentage"],
                           results[town]["duration"], ])
        tot_collisions += results[town]["collisions"]
        tot_completion_percentage += results[town]["completion_percentage"]
        tot_duration += results[town]["duration"]
    total_table.append([tot_collisions / len(results),
                        tot_completion_percentage / len(results),
                        tot_duration / len(results), ])
    # (2) LET'S PRINT ALL
    to_print = "# Singular City Results #"
    print(utils.color_info_string("#" * len(to_print)))
    print(utils.color_info_string(to_print))
    print(utils.color_info_string("#" * len(to_print)))
    print(tabulate(city_table, headers=city_table_head, tablefmt="grid"))
    to_print = "# Total Results #"
    print(utils.color_info_string("#" * len(to_print)))
    print(utils.color_info_string(to_print))
    print(utils.color_info_string("#" * len(to_print)))
    print(tabulate(total_table, headers=total_table_head, tablefmt="grid"))


def parse_jsons(jsons_list_with_absolute_path):
    tot_route_completion = 0
    tot_collisions = 0
    tot_time = 0
    for json in jsons_list_with_absolute_path:
        results = parse_json(json)
        tot_route_completion += results["criteria"]["RouteCompletionTest"]["actual"]
        tot_collisions += results["criteria"]["CollisionTest"]["actual"]
        tot_time += results["criteria"]["Duration"]["actual"]
    n = float(len(jsons_list_with_absolute_path))
    return tot_route_completion/n, tot_collisions/n, tot_time/n


def print_result_per_net(result_per_net_1):
    city_table_head = ["", "Completion", "Collisions", "Time"]
    city_table = []
    for city in result_per_net_1:
        city_table.append([city, result_per_net_1[city][0], result_per_net_1[city][1], result_per_net_1[city][2],])
    print(tabulate(city_table, headers=city_table_head, tablefmt="grid"))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--results_folder",
        help="Folder containing a subfolder for each network that need to be evaluated!",
        required=True,
        type=str
    )
    args = argparser.parse_args()

    evaluation_root_folder = args.results_folder
    networks = os.listdir(evaluation_root_folder)
    all_json = {}
    total_all_json = {}
    result_per_net = {}
    total_result_per_net = {}
    for net in networks:
        cities = os.listdir(os.path.join(evaluation_root_folder, net))
        cities = sorted(cities)
        all_json[f"{net}"] = {}
        total_all_json[f"{net}"] = []
        result_per_net[f"{net}"] = {}
        total_result_per_net[f"{net}"] = {}
        for city in cities:
            routes = os.listdir(os.path.join(evaluation_root_folder, net, city))
            all_json[f"{net}"][city] = []
            for route in routes:
                all_json[f"{net}"][city].append(os.path.join(evaluation_root_folder, net, city, route))
                total_all_json[f"{net}"].append(os.path.join(evaluation_root_folder, net, city, route))
            net_total_results = parse_jsons(all_json[f"{net}"][city])
            result_per_net[f"{net}"][city] = net_total_results
        total_net_total_results = parse_jsons(total_all_json[f"{net}"])
        total_result_per_net[f"{net}"] = total_net_total_results
    print(total_result_per_net)
    print("@"*200)
    for net in networks:
        print("@"*(len(net)+2))
        print(f"@{net}@")
        print("@" * (len(net) + 2))
        print_result_per_net(result_per_net[net])
