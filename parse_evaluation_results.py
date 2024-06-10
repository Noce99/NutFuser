import json
import os
import pathlib
from datetime import datetime
from tabulate import tabulate
from nutfuser import utils

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
        after_underscore = a_json.split("_")[1]         # 92024-06-05-23-55-10.json
        before_minus = after_underscore.split("-")[0]   # 92024
        scenario_num = before_minus[:-4]                # 9
        # (2) LET'S GET THE DATE
        before_dot = after_underscore.split(".")[0]     # 92024-06-05-23-55-10
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
                                                 "flow":    os.path.join(path_to_append, a_json)}
            else:
                # other is FLOW
                json_in_couples[scenario_num] = {"NO_flow": os.path.join(path_to_append, a_json),
                                                 "flow":    os.path.join(path_to_append, other_json_name)}
    return json_in_couples

def count_collisions_and_completion_percentage(json_in_couples):
    total_collisions =              {"NO_flow":0,   "flow":0}
    total_completion_percentage =   {"NO_flow":0.0, "flow":0.0}
    total_duration =                {"NO_flow":0.0, "flow":0.0}
    for scenarios in json_in_couples:
        no_flow_json =  json_in_couples[scenarios]["NO_flow"]
        flow_json =     json_in_couples[scenarios]["flow"]
        parsed_no_flow =    parse_json(no_flow_json)
        parsed_flow =       parse_json(flow_json)
        assert parsed_no_flow   ["scenario_num"] == scenarios
        assert parsed_flow      ["scenario_num"] == scenarios
        for criteria in SELECTED_CRITERIA:
            assert parsed_no_flow["criteria"]   [criteria] is not None
            assert parsed_flow["criteria"]      [criteria] is not None
        total_collisions["NO_flow"] +=              parsed_no_flow  ["criteria"]["CollisionTest"]["actual"]
        total_collisions["flow"] +=                 parsed_flow     ["criteria"]["CollisionTest"]["actual"]
        total_completion_percentage["NO_flow"] +=   parsed_no_flow  ["criteria"]["RouteCompletionTest"]["actual"]
        total_completion_percentage["flow"] +=      parsed_flow     ["criteria"]["RouteCompletionTest"]["actual"]
        total_duration["NO_flow"] +=                parsed_no_flow  ["criteria"]["Duration"]["actual"]
        total_duration["flow"] +=                   parsed_flow     ["criteria"]["Duration"]["actual"]

        if abs(parsed_no_flow["criteria"]["CollisionTest"]["actual"] - parsed_flow["criteria"]["CollisionTest"]["actual"]) > 3:
            print(f'Interesting one! ({scenarios}) FLOW/NO_FLOW = {parsed_flow["criteria"]["CollisionTest"]["actual"]}/{parsed_no_flow["criteria"]["CollisionTest"]["actual"]}')
    
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

def parse_json(a_json):
    result = {}
    with open(a_json, 'r') as f:
        data = json.load(f)
        scenario_num = data["scenario"][len("RouteScenario_"):]
        my_criteria = {element:None for element in SELECTED_CRITERIA}
        criteria = data["criteria"]
        for a_criteria in criteria:
            name = a_criteria["name"]
            expected = a_criteria["expected"]
            actual = a_criteria["actual"]
            if name in SELECTED_CRITERIA:
                my_criteria[name] = {"expected":expected, "actual":actual}
        result["scenario_num"] = scenario_num
        result["criteria"] = my_criteria
    return result

def print_results(results):
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
        city_table.append([ town.split("_")[0],
                            results[town]["num_of_routes"],
                            results[town]["collisions"]             ["NO_flow"],
                            results[town]["collisions"]             ["flow"],
                            results[town]["completion_percentage"]  ["NO_flow"],
                            results[town]["completion_percentage"]  ["flow"],
                            results[town]["duration"]               ["NO_flow"],
                            results[town]["duration"]               ["flow"],])
        tot_collisions_no_flow              += results[town]["collisions"]             ["NO_flow"]
        tot_collisions_flow                 += results[town]["collisions"]             ["flow"]
        tot_completion_percentage_no_flow   += results[town]["completion_percentage"]  ["NO_flow"]
        tot_completion_percentage_flow      += results[town]["completion_percentage"]  ["flow"]
        tot_duration_no_flow                += results[town]["duration"]               ["NO_flow"]
        tot_duration_flow                   += results[town]["duration"]               ["flow"]
    total_table.append([tot_collisions_no_flow / len(results),
                        tot_collisions_flow / len(results),
                        tot_completion_percentage_no_flow / len(results),
                        tot_completion_percentage_flow / len(results),
                        tot_duration_no_flow / len(results),
                        tot_duration_flow / len(results)])
    # (2) LET'S PRINT ALL
    to_print = "# Singular City Results #"
    print(utils.color_info_string("#"*len(to_print)))
    print(utils.color_info_string(to_print))
    print(utils.color_info_string("#"*len(to_print)))
    print(tabulate(city_table, headers=city_table_head, tablefmt="grid"))
    to_print = "# Total Results #"
    print(utils.color_info_string("#"*len(to_print)))
    print(utils.color_info_string(to_print))
    print(utils.color_info_string("#"*len(to_print)))
    print(tabulate(total_table, headers=total_table_head, tablefmt="grid"))

if __name__ == "__main__":
    evaluation_root_folder = os.path.join(pathlib.Path(__file__).parent.resolve(), "evaluation")
    evaluations = os.listdir(evaluation_root_folder)
    evaluations.remove("not_import_results")
    if len(evaluations) > 1:
        print("Something fancy for selecting witch evaluation folder to choose!")
        
        print(f"For now I choose: {evaluations[0]}")
        selected_evaluation_folder = evaluations[0]
    else:
        selected_evaluation_folder = evaluations[0]
    
    all_evaluation_sub_folder = os.listdir(os.path.join(evaluation_root_folder, selected_evaluation_folder))

    results = {}
    total_simulation_time = 0
    for sub_folder in all_evaluation_sub_folder:
        current_full_parent_path = os.path.join(evaluation_root_folder, selected_evaluation_folder, sub_folder)
        all_json = os.listdir(current_full_parent_path)
        json_in_couples = put_jsons_in_couple(all_json, current_full_parent_path)
        print(sub_folder)
        collisions, completion_percentage, \
            duration, simulation_time = count_collisions_and_completion_percentage(json_in_couples)
        results[sub_folder] = {"collisions"             :collisions,
                               "completion_percentage"  :completion_percentage,
                               "duration"               :duration,
                               "num_of_routes"          :len(json_in_couples)}
        total_simulation_time += simulation_time

    to_print = "# SUM UP #"
    print(utils.color_info_string("#"*len(to_print)))
    print(utils.color_info_string(to_print))
    print(utils.color_info_string("#"*len(to_print)))
    print(utils.color_info_success(f"Found out a total of {total_simulation_time/3600:.2f} hours of simulation time in {len(all_evaluation_sub_folder)} towns."))
    print(utils.color_info_success(f"Each entry in the table should be consider as a per route value."))
    print_results(results)

        