import random

def player(prev_play, opponent_history=[]):
    # Check if it's the first game
    if prev_play == "":
        return random.choice(["R", "P", "S"])

    # Update opponent's history
    opponent_history.append(prev_play)

    # Implement your winning strategy based on opponent's history
    # This is just a simple example, you need to develop a better strategy
    if opponent_history.count("R") > opponent_history.count("P"):
        return "P"
    elif opponent_history.count("P") > opponent_history.count("S"):
        return "S"
    else:
        return "R"

# Testing the player against a bot (e.g., quincy) for 1000 games
# play(player, quincy, 1000, verbose=True)
