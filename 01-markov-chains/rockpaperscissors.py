import random
import matplotlib.pyplot as plt

MOVES = ["Rock", "Paper", "Scissors"]

INITIAL_TRANSITION_MATRIX = {
    "Paper": {"Paper": 2/3, "Rock": 1/3, "Scissors": 0},
    "Rock": {"Paper": 0, "Rock": 2/3, "Scissors": 1/3},
    "Scissors": {"Paper": 2/3, "Rock": 0, "Scissors": 1/3}
}


def copy_matrix(matrix):
    return {row: matrix[row].copy() for row in matrix}


def get_winning_move(move):
    if move == "Rock":
        return "Paper"
    elif move == "Paper":
        return "Scissors"
    else:
        return "Rock"


def get_reward(move1, move2):

    if move1 == move2:
        return 0
    elif (
        (move1 == "Rock" and move2 == "Scissors") or
        (move1 == "Paper" and move2 == "Rock") or
        (move1 == "Scissors" and move2 == "Paper")
    ):
        return 1
    else:
        return -1


def predict_next_move(matrix, last_opponent_move):

    row = matrix[last_opponent_move]
    return max(row, key=row.get)


def static_player_move(last_opponent_move, static_matrix):

    predicted_move = predict_next_move(static_matrix, last_opponent_move)
    return get_winning_move(predicted_move)


def learning_player_move(last_opponent_move, learning_matrix):

    predicted_move = predict_next_move(learning_matrix, last_opponent_move)
    return get_winning_move(predicted_move)


def update_learning_counts(counts_matrix, previous_move, current_move):

    counts_matrix[previous_move][current_move] += 1


def normalize_counts_to_probabilities(counts_matrix):

    probability_matrix = {}

    for prev_move in counts_matrix:
        total = sum(counts_matrix[prev_move].values())
        probability_matrix[prev_move] = {}

        for next_move in counts_matrix[prev_move]:
            probability_matrix[prev_move][next_move] = counts_matrix[prev_move][next_move] / total

    return probability_matrix


def simulate_games(num_games=10000):

    static_matrix = copy_matrix(INITIAL_TRANSITION_MATRIX)


    learning_counts = {
        "Paper": {"Paper": 2, "Rock": 1, "Scissors": 0},
        "Rock": {"Paper": 0, "Rock": 2, "Scissors": 1},
        "Scissors": {"Paper": 2, "Rock": 0, "Scissors": 1}
    }
    learning_matrix = normalize_counts_to_probabilities(learning_counts)

    last_move_of_static = random.choice(MOVES)
    last_move_of_learning = random.choice(MOVES)

    static_total_reward = 0
    learning_total_reward = 0

    static_rewards_over_time = []
    learning_rewards_over_time = []

    for _ in range(num_games):
        static_move = static_player_move(last_move_of_learning, static_matrix)
        learning_move = learning_player_move(last_move_of_static, learning_matrix)


        static_reward = get_reward(static_move, learning_move)
        learning_reward = get_reward(learning_move, static_move)

        static_total_reward += static_reward
        learning_total_reward += learning_reward

        static_rewards_over_time.append(static_total_reward)
        learning_rewards_over_time.append(learning_total_reward)


        update_learning_counts(learning_counts, last_move_of_static, static_move)
        learning_matrix = normalize_counts_to_probabilities(learning_counts)


        last_move_of_static = static_move
        last_move_of_learning = learning_move

    return static_rewards_over_time, learning_rewards_over_time


def plot_results(static_rewards, learning_rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(static_rewards, label="Static Model")
    plt.plot(learning_rewards, label="Learning Model")
    plt.xlabel("Games")
    plt.ylabel("Total Reward")
    plt.title("Static vs Learning Model")
    plt.legend()
    plt.grid(True)
    plt.show()


def print_analysis(static_rewards, learning_rewards):
    final_static = static_rewards[-1]
    final_learning = learning_rewards[-1]

    print("Final total reward of Static Model:", final_static)
    print("Final total reward of Learning Model:", final_learning)
    print()

    if final_learning > final_static:
        print("Analysis:")
        print("The learning model achieved a higher total reward over time.")
        print("This suggests that updating the transition matrix helped the model adapt to the opponent's behavior.")
    elif final_learning < final_static:
        print("Analysis:")
        print("The static model achieved a higher total reward in this simulation.")
        print("This suggests that the learning model did not adapt efficiently enough against the opponent's strategy.")
    else:
        print("Analysis:")
        print("Both models achieved similar total rewards.")
        print("This suggests that neither approach had a strong long-term advantage in this simulation.")


def main():
    num_games = 10000
    static_rewards, learning_rewards = simulate_games(num_games)
    plot_results(static_rewards, learning_rewards)
    print_analysis(static_rewards, learning_rewards)


if __name__ == "__main__":
    main()