import time
import os
import pyautogui
import numpy as np

from tools.utils import encode_image, log_output, get_annotate_img
from tools.serving.api_providers import anthropic_completion, anthropic_text_completion, openai_completion, openai_text_reasoning_completion, gemini_completion, gemini_text_completion, deepseek_text_reasoning_completion
import re
import json

CACHE_DIR = "cache/sokoban"

def load_game_state(filename='game_state.json'):
    filename = os.path.join(CACHE_DIR, filename)
    """Load the game matrix from a JSON file."""
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading matrix: {e}")
        return None

def convert_to_text_table_and_matrix(game_state):
    """Convert a 2D list matrix into a structured text table and transform the matrix based on item_map."""
    header = "ID  | Item Type    | Position"
    line_separator = "-" * len(header)

    item_map = {
        '#': 'Wall',
        '@': 'Worker',
        '$': 'Box',
        '?': 'Dock',
        '*': 'Box on Dock',
        ' ': 'Empty'
    }

    table_rows = [header, line_separator]
    matrix = []
    item_id = 1

    for row_idx, row in enumerate(game_state):
        transformed_row = []
        for col_idx, cell in enumerate(row):
            item_type = item_map.get(cell, 'Unknown')
            table_rows.append(f"{item_id:<3} | {item_type:<12} | ({row_idx}, {col_idx})")
            transformed_row.append(f'({row_idx}, {col_idx}) {item_type}')
            # transformed_row.append(item_type)
            item_id += 1
        matrix.append('| '.join(transformed_row))

    return "\n".join(table_rows), matrix

def matrix_to_string(matrix):
    """Convert a 2D list matrix into a string with each row on a new line."""
    # If each element is already a string or you want a space between them:
    return "\n".join(" ".join(str(cell) for cell in row) for row in matrix)

def matrix_to_string2(matrix):
    """Convert a 2D list matrix into a string with each row on a new line."""
    return "\n".join(("[" + (",".join(str(cell) for cell in row)) + "]") for row in matrix)


def log_move_and_thought(move, thought, latency):
    """
    Logs the move and thought process into a log file inside the cache directory.
    """
    log_file_path = os.path.join(CACHE_DIR, "sokoban_moves.log")
    
    log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Move: {move}, Thought: {thought}, Latency: {latency:.2f} sec\n"
    
    try:
        with open(log_file_path, "a") as log_file:
            log_file.write(log_entry)
    except Exception as e:
        print(f"[ERROR] Failed to write log entry: {e}")

def call_llm(api_provider, system_prompt, model_name, prompt, thinking=True, base64_image=None, modality="vision-text"):
    """
    Calls the appropriate LLM API based on the provider, model, and modality.
    """
    if api_provider == "anthropic" and modality == "text-only":
        return anthropic_text_completion(system_prompt, model_name, prompt, thinking)
    elif api_provider == "anthropic":
        return anthropic_completion(system_prompt, model_name, base64_image, prompt, thinking)
    elif api_provider == "openai" and "o3" in model_name and modality == "text-only":
        return openai_text_reasoning_completion(system_prompt, model_name, prompt)
    elif api_provider == "openai":
        return openai_completion(system_prompt, model_name, base64_image, prompt)
    elif api_provider == "gemini" and modality == "text-only":
        return gemini_text_completion(system_prompt, model_name, prompt)
    elif api_provider == "gemini":
        return gemini_completion(system_prompt, model_name, base64_image, prompt)
    elif api_provider == "deepseek":
        return deepseek_text_reasoning_completion(system_prompt, model_name, prompt)
    else:
        raise NotImplementedError(f"API provider: {api_provider} is not supported.")

def sokoban_read_worker(system_prompt, api_provider, model_name, image_path):
    # what's the point of base64_image here?
    base64_image = encode_image(image_path)

    game_state = load_game_state()
    if game_state is not None:
        board_str, matrix = convert_to_text_table_and_matrix(game_state)
    else:
        board_str = "No board available."
    return board_str, matrix

SOKOBAN_GAME_RULES = '''
## Sokoban Game Rules
- The Sokoban board is structured as a matrix of items with coordinated positions: (row_index, column_index). Both row_index and column_index start with 0.
- Each item in the board matrix can be Wall, Box, Dock, Empty, Worker, Worker on Dock, or Box on Dock. There is only one worker in the board.
    - 'Wall': Wall cannot be moved or crossed. It is a solid barrier.
    - 'Worker': The player-controlled character. The worker can move in four directions (up, down, left, right) but cannot move diagonally.
    - "Box": Box is a movable item that the worker can push. The worker can only push one box at a time.
    - "Dock": Dock is a target location for boxes. When a box reaches a dock location, it is marked as a Box on Dock.    
    - "Worker on Dock": When the worker reaches a dock location, it is marked as a Worker on Dock. The worker on dock can move away from the dock.
    - 'Box on Dock': Box on dock is a box that has been successfully pushed to a dock location. It can also be pushed away from the dock.
    - ' ': Empty space. Worker can move to this space. Box can be pushed to this space.
- You control the worker who can move in four directions (up decrements row_index, down increments row_index, left decrements column_index, right decrements column_index) in the 2D Sokoban game board. The worker cannot move diagonally in one step.
- You can push boxes if the worker is positioned next to the box and the opposite side of the box is empty.
- You can not push the box into a wall or another box.
- The worker can only push one box at a time.
- The worker can not pull boxes.
- The worker can not move through walls or boxes.
- When a box reaches a dock location, it is marked as a box on dock. A box on dock can also be pushed away from the dock.
- When the worker reaches a dock location, it is marked as a Worker on Dock. The worker on dock can move away from the dock.
- The goals is to push all boxes onto the dock locations.
- When all boxes reaches dock location, the current level is completed. Then a new level is started. The board will be refreshed; and you should start over by ignore all previous thoughts with action 'cleanup'. 
'''

METHODOLOGY_AND_TIPS = '''
## Methodology of playing Sokoban
- Make a plan of moving boxes to the dock locations.
- The plan should contain the next multiple steps by considering all possible paths for each box, ensuring they will have a viable step-by-step path to reach their dock locations.
- Before leaving a box. Consider if it will become a road block for future boxes.
- Consider relationship among boxes, you can run the Rolling Stone algorithm: Iterative Deepening A* (IDA*) algorithm to find an optimal path.
- Sometimes, you may need to push box to opposite direction to make space for the box to be pushed to the dock location.
- Start with the end. Think how the last box will reach the dock location. Then work backward to find the path for the worker.
- Before making a move, re-analyze the entire puzzle layout.
- Do not waste too much time thinking. If you are not sure, just make a move. You can always unmove or restart to compose a new plan.
- Always identify critical locations in the board. Critical locations are box or empty items that are in the passages connecting rooms. You should plan moves across the critical locations instead of wasting time in hitting walls.

## Tips
- Every level is designed to be solvable. If you are stuck, it is likely that you have made a mistake in your plan. You can use the 'unmove' action to undo the last move and try again.
- You can use the 'restart' action to restart the current level if you get stuck.
- You might need to move around some boxes or walls to create space for the worker to move.
- If you spend more than 5 steps in one level without any progress, consider to push one box to the opposite direction to make space for the box to be pushed to the dock location.
- If the worker is stuck in a closed room with boxes, you can push the box to the opposite direction to make space for the worker to move.
- Every empty space could be used. If you have a box blocking way to some space, you can push the box to towards the space and then push it back.
- If the worker is between the box and the dock, the worker should move around to make the box between the worker and the dock. During the moving, it is totally OK to push the box around in order to make it between the workder and a dock.
- If the worker to a target location is blocked by walls and boxes, you can push the box to the target location and then move the worker to the target location.
'''

#"- If the board has two rooms connected by a narrow passage, you should try a plan to push one box to the opposite direction to make space for the box to be pushed back to the dock location."


def sokoban_planner(system_prompt, api_provider, model_name, thinking=True, modality="text-only", level=1):
    '''
    A planner agent to generate a list of mutually exclusive and collectively inclusive strategies for the Sokoban game.
    Each strategy will have an ID and will be saved to a file for later retrieval.
    '''

    if system_prompt is None:
        system_prompt = (
            "You are an expert Sokoban planner. Your job is to generate a list of mutually exclusive and collectively inclusive strategies for the Sokoban game."
        )

    # Load the current game state
    game_state = load_game_state()
    if game_state is not None:
        board_str, matrix = convert_to_text_table_and_matrix(game_state)
    else:
        board_str = "No board available."

    prompt = (
        f"{SOKOBAN_GAME_RULES}"
        f"You are currently in level {level}.\n"
        f"{METHODOLOGY_AND_TIPS}"

        "\n## Planning Instructions\n"
        "Analyze the current Sokoban board and compose multiple strategies to move boxes to the dock locations.\n"
        "The output should be a list of no less than 3 strategies.\n"
        "The strategies should be general guidelines to solve current level.\n"
        "The strategies should be mutually exclusive and collectively inclusive.\n"
        "Each strategy should not have speicific moves. Instead, each strategy should have an instructional plan in phases to complete the whole level. \n"
        "Each strategy should have a unique ID.\n"

        "# Board State\n"
        "\nHere is the current layout of the Sokoban board:\n"
        f"{board_str}\n\n"

        "## Output Format\n"
        "Return only the JSON list of strategies as shown above. The response should start with string '[\n'."
        "\nExample Output (JSON):\n"
        "[\n"
        "  {\"id\": 1, \"strategy\": \"Push box 1 to dock 2, then box 2 to dock 1.\"},\n"
        "  {\"id\": 2, \"strategy\": \"Push box 2 to the right, then box 1 to dock 1, then box 2 to dock 2.\"}\n"
        "]\n\n"
    )

    # Call the LLM to get strategies
    response = call_llm(api_provider, system_prompt, model_name, prompt, thinking, None, modality)

    # Try to parse the response as JSON
    try:
        strategies = json.loads(response)

        print(strategies)

        # Ensure each strategy has a unique string ID
        for i, strat in enumerate(strategies):
            if 'id' not in strat:
                strat['id'] = str(i+1)
            else:
                strat['id'] = str(strat['id'])
    except Exception as e:
        print(f"[ERROR] Failed to parse strategies: {e}\nResponse: {response}")
        strategies = []

    # Save strategies to file
    strategies_file = os.path.join(CACHE_DIR, "sokoban_strategies.json")
    try:
        with open(strategies_file, "w") as f:
            json.dump(strategies, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save strategies: {e}")

    return strategies

def sokoban_player(system_prompt, api_provider, model_name, 
    prev_response="", 
    thinking=True, 
    modality="vision-text",
    level=1,
    critic_feedback=None,
    crop_left=0, 
    crop_right=0, 
    crop_top=0, 
    crop_bottom=0, 
    ):
    """
    1) Captures a screenshot of the current game state. (The screenshot is actually generated by game itself)
    2) Calls an LLM to generate PyAutoGUI code for the next move.
    3) Logs latency and the generated code.
    """
    # Capture a screenshot of the current game state.
    # Save the screenshot directly in the cache directory.
    assert modality in ["text-only", "vision-text"], f"modality {modality} is not supported."

    os.makedirs("cache/sokoban", exist_ok=True)
    screenshot_path = "cache/sokoban/sokoban_screenshot.png"

    levels_dim_path = os.path.join(CACHE_DIR, "levels_dim.json")
    with open(levels_dim_path, "r") as f:
        levels_dims = json.load(f)

    # Extract rows/cols for the specified level
    level_key = f"level_{level}"
    if level_key not in levels_dims:
        raise ValueError(f"No dimension info found for {level_key} in {levels_dim_path}")

    grid_rows = levels_dims[level_key]["rows"]
    grid_cols = levels_dims[level_key]["cols"]

    annotate_image_path, grid_annotation_path, annotate_cropped_image_path = get_annotate_img(screenshot_path, crop_left=crop_left, crop_right=crop_right, crop_top=crop_top, crop_bottom=crop_bottom, grid_rows=grid_rows, grid_cols=grid_cols, cache_dir=CACHE_DIR)

    table, matrix = sokoban_read_worker(system_prompt, api_provider, model_name, screenshot_path)

    # print(f"-------------- TABLE --------------\n{table}\n")
    # print(f"-------------- MATRIX --------------\n{matrix_to_string2(matrix)}\n")
    # print(f"-------------- MATRIX --------------\n{matrix}\n")
    #print(f"-------------- prev response --------------\n{prev_response}\n")

    prompt = (
    f"{SOKOBAN_GAME_RULES}"

    f"You are currently in level {level}.\n"

    f"{METHODOLOGY_AND_TIPS}"

    "## Feedback and Critique\n"
    "Another critic provides critique and feedback based on your last step thoughts and moves. It might be helpful to refine your plan. Please take the feedback into consideration. The feedback is:"
    f"{critic_feedback}.\n\n"

    "## Potential Deadlocks to avoid:\n"
    "1. If a box is pushed to a wall, it cannot move away from the wall, unless the box can be pushed along the wall to be away from the wall.\n"
    "2. If a box is pushed to a corner, it cannot move away from the corner.\n"

    f"Here is your previous response: {prev_response}. Please evaluate your plan and thought about whether we should correct or adjust.\n"

    "## Strategy\n"
    "Utilize the main vertical corridor as the primary path for the box. First, push the box downwards into the corridor, potentially sending it into the lower section. Then, maneuver the worker through the larger open area to reposition below the box in the corridor. Finally, push the box upwards along the corridor, step by step, until it reaches the dock.\n"

    "## Sokoban Game board"
    "Here is the current layout of the Sokoban board:\n"
    f"{table}\n\n"
    # "Here is the current layout in 2-dimension array:\n"
    # f"{matrix}\n\n"

    "## Output Format:\n"
    "The output should be on line of text, each line should contain a thought process and a move.\n"
    "The output should be in the following format:\n"
    "<thought>{thought process}</thought><move>{action}</move>\n\n"

    "The thought process should be a detailed and thoughtful plan of steps following the Strategy above. It should:\n"
    "- identify the locations of the worker, boxes, and docks, and walls first. This is to have an overview of the board\n"
    "- identfy passages in the board and critical empty locations between rooms.\n"
    "- Keep refining your plan. Taking the Critic's feedback into consideration, but don't fully trust his idea. You own your plan.\n"
    "- Following methodology and tips mentioned before to compose a plan by list the path to move boxes to docks. For example, (1,1)->(1,2)->(2,3) to move box 1 at (1,2) to doc at (2, 3). Always try to get a complete path before execution.\n" 
    "- Stick to the strategy.\n"   

    "The action should be one of the following"
    "- 'up' decrements the row_index of the worker in board.\n"
    "- 'down' increments the row_index of the worker in board\n"
    "- 'left' decrements the column_index of the worker in board\n"
    "- 'right' increments the column_index of the worker in board\n"
    "- 'restart' means to restart current level.\n"
    "- 'unmove' means to undo the last move.\n"
    "- 'cleanup' means that the current level is completed. Then a new level is started. The board will be refreshed; and you should start over by ignore all previous thoughts. You should only do 'cleanup' if the curret layout of the Sokoban board shows that all boxes are on docks.\n"
    "All action cannot cross wall.\n"
    "Actions up/down/left/right can only push boxes if the worker is positioned next to the box and the opposite side of the box is empty.\n"

    "Example output 1 (single move):"
    "```\n"
    "<thought>"
    # "The board layout is like this:\n"
    # "ID | Item Type    | Position\n"
    # "1  | Wall         | (0, 0)\n"
    # "2  | Wall         | (0, 1)\n"
    # "3  | Wall         | (0, 2)\n"
    # "4  | Wall         | (0, 3)\n"
    # "5  | Wall         | (1, 0)\n"
    # "6  | Dock         | (1, 1)\n"
    # "7  | Empty        | (1, 2)\n"
    # "8  | Wall         | (1, 3)\n"
    # "9  | Wall         | (2, 0)\n"
    # "10 | Worker       | (2, 1)\n"
    # "11 | Empty        | (2, 2)\n"
    # "12 | Wall         | (2, 3)\n"
    # "13 | Wall         | (3, 0)\n"
    # "14 | Box          | (3, 1)\n"
    # "15 | Wall         | (3, 2)\n"
    # "16 | Wall         | (3, 3)\n"
    # "17 | Wall         | (4, 0)\n"
    # "18 | Empty        | (4, 1)\n"
    # "19 | Empty        | (4, 2)\n"
    # "20 | Wall         | (4, 3)\n"
    # "21 | Wall         | (5, 0)\n"
    # "22 | Empty        | (5, 1)\n"
    # "23 | Empty        | (5, 2)\n"
    # "24 | Wall         | (5, 3)\n"
    # "25 | Wall         | (6, 0)\n"
    # "26 | Empty        | (6, 1)\n"
    # "27 | Empty        | (6, 2)\n"
    # "28 | Wall         | (6, 3)\n"
    # "29 | Wall         | (7, 0)\n"
    # "30 | Empty        | (7, 1)\n"
    # "31 | Empty        | (7, 2)\n"
    # "32 | Wall         | (7, 3)\n"
    # "33 | Wall         | (8, 0)\n"
    # "34 | Wall         | (8, 1)\n"
    # "35 | Wall         | (8, 2)\n"
    # "36 | Wall         | (8, 3)\n"
    # "\n"
    "The worker is at (2, 1) and the box is at (3, 1). The dock is at (1, 1). The location (3, 1) is in critical path since it is the passge connecting upper room and downer room.\n"
    "To push the box at (3,1) to dock at (1, 1), the worker has to move to the opposite side of the box. It looks there is no way to move around the box in the narrow space. So, I will push the box to a wider space. Then it is possible to keep the box between the dock and the worker.\n"
    "Plans:\n"
    "Phase 1: Push the box to the right to wider space.\n"
    "Phase 2: Move the worker to make the box between the dock and the worker.\n"
    "Phase 3: Push the box to the dock.\n"
    "Details of Plans:\n"
    "The Phase 1 moves are: (2, 1) -> (3, 1) -> (4, 1). After these moves, the box is supposed to be pushed at (5, 1) after these moves. Both the worker and the box are in a wider space.\n"    
    "The Phase 2 moves are: (4, 1) -> (4, 2) -> (5, 2) -> (6, 2) -> (6, 1). After these moves, the box is still at (5, 1) which is between the worker and the dock.\n"
    "The Phase 3 moves are: (6, 1) -> (5, 1) -> (4, 1) -> (3, 1) -> (2, 1). These moves will push the box to the dock.\n"
    "So, the next step is down to (3, 1).</thought><move>down</move>\n"
    "```\n"
    
    # "Example output 2 (multiple moves):"
    # "<thought>Positioning the worker to access other boxes and docks for future moves. The path of the worker will be (2, 3) -> (2, 4) -> (3, 4).</thought><move>right</move>"
    # "<thought>Positioning the worker to access other boxes and docks for future moves. The path of the worker will be (2, 4) -> (3, 4)</thought><move>up</move>\n\n"
    )


    base64_image = encode_image(annotate_cropped_image_path)
    if "o3-mini" in model_name:
        base64_image = None
    start_time = time.time()

    print(f"Calling {model_name} API...")
    # Call the LLM API based on the selected provider.
    response = call_llm(api_provider, system_prompt, model_name, prompt, thinking, base64_image, modality)

    latency = time.time() - start_time

    pattern = r'<thought>([^<]*)</thought><move>(\w+)</move>'
    matches = re.findall(pattern, response, re.IGNORECASE)

    move_thought_list = []
    # Loop through every move in the order they appear
    for thought, move in matches:
        move = move.strip().lower()
        thought = thought.strip()

        action_pair = {"move": move, "thought": thought}
        move_thought_list.append(action_pair)

        # Log move and thought
        log_output(
            "sokoban_player",
            f"[INFO] Move executed: ({move}) | Thought: {thought} | Latency: {latency:.2f} sec",
            "sokoban",
            mode="a",
        )

    # response
    return move_thought_list

def sokoban_critic(moves_thoughts=None, last_action=None, system_prompt=None, api_provider=None, model_name=None, thinking=True, modality="text-only", level=1):
    """
    A critic agent to evaluate the moves and plans provided by sokoban_player.
    Uses an LLM to provide constructive suggestions based on the overall game state and Sokoban rules.
    Args:
        moves_thoughts: List of dicts, each with 'move' and 'thought' from sokoban_player.
        game_state: Optional. 2D list representing the current board. If None, loads from cache.
        last_action: The last move issued by sokoban_player (string).
        system_prompt, api_provider, model_name, thinking, modality, level: LLM config, same as sokoban_player.
    Returns:
        LLM-generated feedback string.
    """
    game_state = load_game_state()
    if system_prompt is None:
        system_prompt = (
            "You are an expert Sokoban critic. Your job is to evaluate the quality, safety, and optimality of a sequence of moves and plans for Sokoban, following all Sokoban rules and best practices."
            "You must point out any mistakes, risks (such as deadlocks), inefficiencies, or missed opportunities for improvement. "
            "Be constructive and specific. If the plan is good, explain why. If not, suggest concrete improvements."
            "You should be creative and think outside the box. When the plan makes no progress after multiple steps, you can suggest a random push to change the situation."
        )

    # Prepare board state as text
    if game_state is not None:
        # board_str = matrix_to_string(game_state)
        board_str, matrix = convert_to_text_table_and_matrix(game_state)
    else:
        board_str = "No board available."

    # Prepare moves/thoughts as text
    last_action_str = f"The last action issued was: {last_action}." if last_action else ""

    # Sophisticated prompt for LLM
    prompt = (
        f"{SOKOBAN_GAME_RULES}"

        f"{METHODOLOGY_AND_TIPS}"

        f"You are currently in level {level}.\n"
    
        "## Strategy\n"
        "Utilize the main vertical corridor as the primary path for the box. First, push the box downwards into the corridor, potentially sending it into the lower section. Then, maneuver the worker through the larger open area to reposition below the box in the corridor. Finally, push the box upwards along the corridor, step by step, until it reaches the dock.\n"


        "## Sokoban Critic Instructions\n"
        "You are given a Sokoban board state and a sequence of moves and thoughts by another player that lead to the current board state.\n"
        "You are also given a Strategy that guides the player."
        "Your job is to critically evaluate the plan and moves, following these guidelines:\n"
        "- Evaluate whether the moves are safe and follow the Strategy. Ensure the player's plan stick to the Strategy.\n"
        # "- Evaluate whether given thoughts and plans would end in deadlocks or repeated/looping actions or crossing walls.\n"
        "- Identify if any move risks pushing a box into a corner or against a wall where it cannot be recovered.\n"
        "- Assess if the plan is efficient and optimal, or if there are unnecessary steps.\n"
        "- If the last action creates a deadlock, please suggest the other player to unmove.\n"
        "- If the plan is good, explain why. If not, suggest specific improvements.\n"
        "- You should keep challenging the player whether his plan consider all possible paths for each box, ensuring they will have a viable step-by-step path to reach their dock locations.\n"
        "- Don't suggeet next move. Just evaluate the plan and suggest improvements.\n"
        # hack for level 5
        "- If the board has two rooms connected by a narrow passage, you should suggest the player to push one box to the opposite direction to make space for the box to be pushed to the dock location.\n"
        "- Always be constructive and detailed.\n\n"

        "## Sokoban Board State (as matrix)\n"
        f"{last_action_str}\n\n"
        f"The board state is: {board_str}\n\n"

        "## Moves and Thoughts\n"
        f"The player's thoughts and moves: {moves_thoughts}\n\n"

        "## Output Format\n"
        "Write a detailed critique on the other player's thoughts and plans.\n"
        "Acknowlege the  move if the plan is feasible.\n"
        "If the plan is not feasible, suggest new strategy.\n"
    )

    # Call LLM for evaluation (like sokoban_player)
    if api_provider is None or model_name is None:
        raise ValueError("api_provider and model_name must be provided for LLM-based critic.")
    response = call_llm(api_provider, system_prompt, model_name, prompt, thinking, None, modality)

    return response