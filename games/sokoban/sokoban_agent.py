import time
import numpy as np
import concurrent.futures
import argparse
from collections import deque, Counter

import os
import json
import re
import pyautogui
from tools.utils import str2bool

from games.sokoban.workers import sokoban_player, sokoban_critic

CACHE_DIR = "cache/sokoban"

from collections import Counter

def majority_vote_move(moves_list, prev_move=None):
    """
    Returns the majority-voted move from moves_list.
    If there's a tie for the top count, and if prev_move is among those tied moves,
    prev_move is chosen. Otherwise, pick the first move from the tie.
    """
    if not moves_list:
        return None

    c = Counter(moves_list)
    
    # c.most_common() -> list of (move, count) sorted by count descending, then by move
    counts = c.most_common()
    top_count = counts[0][1]  # highest vote count

    tie_moves = [m for m, cnt in counts if cnt == top_count]

    if len(tie_moves) > 1 and prev_move:
        if prev_move in tie_moves:
            return prev_move
        else:
            return tie_moves[0]
    else:
        return tie_moves[0]


# System prompt remains constant
system_prompt = (
    "You are an expert AI agent specialized in solving Sokoban puzzles optimally. "
    "Your goal is to push all boxes onto the designated dock locations while avoiding deadlocks. "
)

def pyautogui_move_handler(move):
    pyautogui.press(move)

def main_loop(api_provider, model_name, modality, thinking, num_threads, no_critic= False, move_handler=pyautogui_move_handler):
    # TODO: enlarge this cache size and clear it when level up
    prev_responses = deque(maxlen=50)
    level = None
    step_count = 0
    critic_feedback = None

    def perform_move(action):
        key_map = {
            "up": "up",
            "down": "down",
            "left": "left",
            "right": "right",
            "restart": 'R',
            "unmove": 'D',
            "levelup": ' ', # do nothing
        }
        if move in key_map:
            move_handler(key_map[move])
            print(f"Performed move: {move}")
        else:
            print(f"[WARNING] Invalid move: {move}")

    try:
        while True:
            current_level_path = os.path.join(CACHE_DIR, "current_level.json")
            with open(current_level_path, 'r') as f:
                level_dict = json.load(f)
                new_level = level_dict["level"]
            
            if new_level != level:
                level = new_level
                # step_count = 0
                new_level_detected = True
                prev_responses.clear()
                print(f"[INFO] New level detected: {level}")
            else:
                new_level_detected = False

            start_time = time.time()

            print(f"level={level}, step={step_count}\n")
            # ------------------------- critic ------------------------ #
            if (step_count > 0 and no_critic == False):
                critic_feedback = sokoban_critic(
                    # moves_thoughts = "\n".join(prev_responses),
                    moves_thoughts = prev_responses[-1] if prev_responses else "No previous messages.",
                    last_action = final_moves[-1] if final_moves else "unknown action",
                    system_prompt = None,
                    api_provider = api_provider,
                    model_name = model_name,
                    thinking = str2bool(thinking),
                    modality = modality,
                    level = level,
                )
                print(f"[INFO] Critic feedback on step {step_count}:")
                print(
                    "------------------critic feedback -------------\n\n",
                    f"{critic_feedback}\n"
                    "-----------------------------------------------\n\n"
                )
                # critic_feedback = "The strategy should be: 1. push the box down towards the bottom of the map. 2. After moving through the corridor to the bottom space, the worker can push the box up towards the top of the map."

            # Self-consistency launch, to disable, set "--num_threads 1"
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for _ in range(num_threads):
                    futures.append(
                        executor.submit(
                            sokoban_player,
                            system_prompt,
                            api_provider,
                            model_name,
                            "\n".join(prev_responses),
                            thinking=str2bool(thinking),
                            modality=modality,
                            level=level,
                            critic_feedback = critic_feedback
                        )
                    )

                # Wait until all threads finish
                concurrent.futures.wait(futures)
                results = [f.result() for f in futures]

            print("all threads finished execution...")
            # print(results)

            # Find the shortest solution length among all threads
            shortest_length = min(len(mlist) for mlist in results)

            # ------------------------- action ------------------------ #
            # For each position up to that shortest length, do a majority vote
            final_moves = []
            collected_thoughts_per_move = []
            # Iterate over all possible future steps
            for i in range(shortest_length):
                # Collect the i-th move and thought from each thread (with sufficient actions predicted)
                move_thought_pairs = [sol[i] for sol in results if len(sol) > i]

                # Vote
                move_candidates = [pair["move"] for pair in move_thought_pairs]
                move_candidate_count = {}
                for move_candidate in move_candidates:
                    if move_candidate in move_candidate_count.keys():
                        move_candidate_count[move_candidate] += 1
                    else:
                        move_candidate_count[move_candidate] = 1

                # print(move_candidate_count)

                if final_moves:
                    chosen_move = majority_vote_move(move_candidates, final_moves[-1])
                else:
                    chosen_move = majority_vote_move(move_candidates)
                final_moves.append(chosen_move)

                # Iterate over all valid threads for this step
                # Gather all thoughts from the threads whose move == chosen_move
                matched_thoughts = [pair["thought"] for pair in move_thought_pairs 
                                     if pair["move"] == chosen_move]

                matched_thought = matched_thoughts[0]

                collected_thoughts_per_move.append(matched_thought)

            # Loop through every move in the order they appear
            for move in final_moves:
                move = move.strip().lower()

                # Perform the move
                perform_move(move)
            # ------------------------- action ------------------------ #

            # HACK: temporary memory module
            if final_moves:
                assert len(final_moves) == len(collected_thoughts_per_move), "move and thought length disagree, regex operation errored out."
                for move, matched_thought in zip(final_moves, collected_thoughts_per_move):
                    step_count += 1
                    latest_response = f"step {step_count} executed:\n" + f"move: {move}, thought: {matched_thought}" + "\n"
                    prev_responses.append(latest_response)

            #TODO: add a critic agent to evaluate the moves
            

            print("[debug] previous message:")
            print("########### Player Thoughts & Moves ############### :\n")
            # print("\n".join(prev_responses))
            print(prev_responses[-1] if prev_responses else "No previous messages.")
            print("###################################################\n\n")
            
            elapsed_time = time.time() - start_time
            time.sleep(1)
            print(f"[INFO] Move executed in {elapsed_time:.2f} seconds.")
    except KeyboardInterrupt:
        print("\nStopped by user.")

def main(move_handler=pyautogui_move_handler):
    parser = argparse.ArgumentParser(description="sokoban AI Agent")
    parser.add_argument("--api_provider", type=str, default="openai", help="API provider to use.")
    parser.add_argument("--model_name", type=str, default="o3-mini", help="LLM model name.")
    parser.add_argument("--modality", type=str, default="text-only", choices=["text-only", "vision-text"],
                        help="modality used.")
    parser.add_argument("--thinking", type=str, default=True, help="Whether to use deep thinking.")
    parser.add_argument("--num_threads", type=int, default=5, help="Number of parallel threads to launch.")
    args = parser.parse_args()

    main_loop(
        api_provider=args.api_provider,
        model_name=args.model_name,
        modality=args.modality,
        thinking=args.thinking,
        num_threads=args.num_threads,
        move_handler=move_handler
    )

if __name__ == "__main__":
    main()