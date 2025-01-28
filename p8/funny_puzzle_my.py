from typing import List, Tuple, Union, Dict
import heapq
import copy

# Note:
# The test cases might be 8-tile, 6-tile, 5-tile or other variants within 3*3 grids. For example, the goal state
# for 8-tile would be [1, 2, 3, 4, 5, 6, 7, 8, 0], and that for 6-tile would be [1, 2, 3, 4, 5, 6, 0, 0, 0]. You are
# encouraged to design cases beyond 7-tile, and design unsolvable cases for solvability check.
# Be sure to follow the printing format specified above and remove debugging output before submission.


def swap_element(state: List[int], pos1: int, pos2: int) -> List[int]:
    """_summary_
        swap those 2 elements in state
    Args:
        state (List[int]): a state of the puzzle
        pos1 (int): swap indexes position1
        pos2 (int): swap indexes position2

    Returns:
        List[int]:new succ state
    """
    new_state = copy.deepcopy(state) # deepcopy allows no change to the input state
    new_state[pos1], new_state[pos2] = new_state[pos2], new_state[pos1]
    return new_state

def compute_goal_position(state: List[int]) -> Tuple[Dict[str, List[int]], List[int]]:
    """_summary_
        compute goal_position of the goal state
    Args:
        state (List[int]): input state

    Returns:
        Tuple[Dict[str, List[int]], List[int]]: 
            Dict[str, List[int]]: goal_position of the goal state
            List[int]: goal state
    """
    # compute the goal state
    zero_num = 0
    for ele in state:
        if ele == 0:
            zero_num += 1
    goal_state = []
    for goal_ele in range(1, 10-zero_num):
        goal_state.append(goal_ele)
    for goal_ele in range(10-zero_num, 10):
        goal_state.append(0)
    # compute the position dict for goal_state
    goal_position = {value: (i // 3, i % 3) for i, value in enumerate(goal_state)}
    return goal_position, goal_state

def compute_manhattan_distance(state: List[int], goal_position: Dict[str, List[int]]) -> int:
    """_summary_
        compute manhattan distance of each state from the goal goal_position
    Args:
        state (List[int]): input state
        goal_position (Dict[str, List[int]]): goal_position of the goal state

    Returns:
        int: total_distance
    """
    total_distance = 0
    for index, value in enumerate(state):
        if value != 0:
            # the position index
            current_position = (index // 3, index % 3)
            goal_position_value = goal_position[value]
            total_distance += abs(current_position[0] - goal_position_value[0]) + abs(current_position[1] - goal_position_value[1])
    return total_distance

def compute_distance(uniq_state_list: List[int]) -> List[int]:
    """_summary_
        compute manhattan distance of each state from the goal state
    Args:
        uniq_state_list (List[int]): a list which have elements of some states
    Returns:
        List[int]: a list with the manhattan distance of every state
    """
    goal_position = compute_goal_position(uniq_state_list[0])[0]
    distances = []
    # for every state in state list, compute its manhattan distance
    for state in uniq_state_list:
        total_distance = compute_manhattan_distance(state, goal_position)
        distances.append(total_distance)
    return distances

def print_succ(state: List[int]) -> None:
    """_summary_
        given a state of the puzzle, represented as a single list of integers
        with a 0 in the empty spaces, print to the console all of the possible successor states
        We do require that these be printed in a specific order!
        just print like beblow:
        [2, 0, 1, 4, 5, 6, 7, 0, 3] h=5
        [2, 5, 1, 0, 4, 6, 7, 0, 3] h=7
        [2, 5, 1, 4, 0, 6, 0, 7, 3] h=7
        [2, 5, 1, 4, 0, 6, 7, 3, 0] h=7
        [2, 5, 1, 4, 6, 0, 7, 0, 3] h=7
    Args:
        state (List[int]): the state list of the puzzle like [2,5,1,4,0,6,7,0,3]
    """
    succ_unique_sorted = compute_succ(state) # successors
    h_value = compute_distance(succ_unique_sorted) # distances
    for idx in range(len(succ_unique_sorted)):
        print(f"{succ_unique_sorted[idx]} h={h_value[idx]}")

def compute_succ(state: List[int]) -> List[int]:
    """_summary_
         given a state of the puzzle, get its succ
    Args:
        state (List[int]): the state list of the puzzle like [2,5,1,4,0,6,7,0,3]
    Returns:
        List[int]: the list of the init stats's succ
    """
    zero_indexes = [i for i, value in enumerate(state) if value == 0]
    succ_without_sort = []
    for zero_index in zero_indexes:
        row, col = zero_index // 3, zero_index % 3 # position of zero
        if row > 0:
            succ_without_sort.append(swap_element(state, zero_index, zero_index - 3))
        if row < 2:
            succ_without_sort.append(swap_element(state, zero_index, zero_index + 3))
        if col > 0:
            succ_without_sort.append(swap_element(state, zero_index, zero_index - 1))
        if col < 2:
            succ_without_sort.append(swap_element(state, zero_index, zero_index + 1))
    succ_unique_sorted = sorted(list(set(tuple(arr) for arr in succ_without_sort)))
    if tuple(state) in succ_unique_sorted:
        succ_unique_sorted.remove(tuple(state))
    succ_unique_sorted = [list(_) for _ in succ_unique_sorted]
    return succ_unique_sorted


def solve(initial_state: List[int]) -> Union[List[List[int]], List]:
    """_summary_
        given a state of the puzzle, perform the A* search algorithm and print the
        path from the current state to the goal state.
        If the puzzle is not solvable, the function should only print(False).
        If the puzzle is solvable, the function should print(True), then print like below:
        ------------
        True
        [4, 3, 0, 5, 1, 6, 7, 2, 0] h=7 moves: 0
        [4, 0, 3, 5, 1, 6, 7, 2, 0] h=6 moves: 1
        [4, 1, 3, 5, 0, 6, 7, 2, 0] h=5 moves: 2
        [4, 1, 3, 0, 5, 6, 7, 2, 0] h=4 moves: 3
        [0, 1, 3, 4, 5, 6, 7, 2, 0] h=3 moves: 4
        [0, 1, 3, 4, 5, 0, 7, 2, 6] h=4 moves: 5
        [0, 1, 3, 4, 0, 5, 7, 2, 6] h=5 moves: 6
        [0, 1, 3, 4, 2, 5, 7, 0, 6] h=4 moves: 7
        [1, 0, 3, 4, 2, 5, 7, 0, 6] h=3 moves: 8
        [1, 2, 3, 4, 0, 5, 7, 0, 6] h=2 moves: 9
        [1, 2, 3, 4, 5, 0, 7, 0, 6] h=1 moves: 10
        [1, 2, 3, 4, 5, 6, 7, 0, 0] h=0 moves: 11
        Max queue length: 163
        -----------
        use heapq.heappush(pq ,(cost, state, (g, h, parent_index)))
    Args:
        initial_state (List[int]): the state list of the puzzle like [2,5,1,4,0,6,7,0,3]

    Returns:
        Union[List[List[int]], List]: state paths
    """
    goal_position, goal_state = compute_goal_position(initial_state)
    pq = []
    history = []
    visited = set()
    initial_h = compute_manhattan_distance(initial_state, goal_position)
    heapq.heappush(pq, (initial_h, initial_state, (0, initial_h, -1)))
    visited.add(tuple(initial_state)) # reduce the complexity, ignore the visited node
    history.append((initial_state, -1)) # use history to build the tree, like the replationship between parent and children
    max_queue_length = 0
    history_path = []
    while pq:
        if len(pq) > max_queue_length:
            max_queue_length = len(pq)
        cost, state, (g, h, parent_index) = heapq.heappop(pq) # pop the minimum cost one
        # print("state: ", state)
        history_path.append(state)
        if state == goal_state:
            path = reconstruct_path(history, len(history) - 1, initial_state)
            print(True)
            for i, (s, idx) in enumerate(path):
                h_value = compute_manhattan_distance(s, goal_position)
                print(f"{s} h={h_value} moves: {i}")
            print(f"Max queue length: {max_queue_length}")
            # print("history: ")
            # for k in history_path:
            #     print(k)
            return [state for state, idx in path]

        succ_states = compute_succ(state)
        # push every successors into the heap
        for succ in succ_states:
            if tuple(succ) not in visited:
                new_g = g + 1
                new_h = compute_manhattan_distance(succ, goal_position)
                new_cost = new_g + new_h
                heapq.heappush(pq, (new_cost, succ, (new_g, new_h, len(history))))
                history.append((succ, parent_index))
                visited.add(tuple(succ))
    print(False)
    return []


def reconstruct_path(history: List[Tuple[List[int], int]], index: int, init_state: List[int]) -> List[List[int]]:
    """_summary_
        build the path from the final state
    Args:
        history (List[Tuple[List[int], int]]): history list
        index (int): the index of the final state in history list
        init_state (List[int]): initial state

    Returns:
        List[List[int]]: the paths list
    """
    path = []
    while index != -1:
        state, parent_index = history[index]
        path.append((state, index))
        index = parent_index
    path.append((init_state, -1))
    path.reverse()
    return path

if __name__ == "__main__":
#     print_succ([4,3,0,5,1,6,7,2,0])
    solve([4,3,0,5,1,6,7,2,0])