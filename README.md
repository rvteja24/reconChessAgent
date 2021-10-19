# reconChessAgent

This repo contains my current work of extending DNNs and Search to Imperfect Information Game of ReconChess.

core.py - contains code for converting board representation to vector representation & for getting the 3x3 grid with max
uncertainty.

MCTSSF.py - Contains code for Monte Carlo Tree Search with StockFish engine backed up by partially trained value net for
estimating leaf node values.

MCTSAgent.py - Contains the code to interact with recon chess engine. Entry point to the agent.

visualizer.py - helps in visualizing the game

Below command conducts match between the agent with one of the baseline bots.

    rc-bot-match MCTSAgent.py otherAgents/troutBot.py --seconds_per_player 300

To connect to the server for playing ranked matches use:

    rc-connect MCTSAgent.py --ranked

For unranked matches remove --ranked tag.