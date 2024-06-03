#!/bin/bash
python ./scenario_runner/scenario_runner.py --route ./scenario_runner/srunner/data/routes_town10_noce.xml --agent ./scenario_runner/srunner/autoagents/nutfuser_autonomous_agent.py --output --timeout 30 --agentConfig /home/enrico/Projects/Carla/NutFuser/train_logs/full_net_NO_flow/model_0030.pth
