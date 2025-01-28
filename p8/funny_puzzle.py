import heapq
from typing import List, Tuple, Union, Dict
import copy

def state_check(state):
    """check the format of state, and return corresponding goal state.
       Do NOT edit this function."""
    non_zero_numbers = [n for n in state if n != 0]
    num_tiles = len(non_zero_numbers)
    if num_tiles == 0:
        raise ValueError('At least one number is not zero.')
    elif num_tiles > 9:
        raise ValueError('At most nine numbers in the state.')
    matched_seq = list(range(1, num_tiles + 1))
    if len(state) != 9 or not all(isinstance(n, int) for n in state):
        raise ValueError('State must be a list contain 9 integers.')
    elif not all(0 <= n <= 9 for n in state):
        raise ValueError('The number in state must be within [0,9].')
    elif len(set(non_zero_numbers)) != len(non_zero_numbers):
        raise ValueError('State can not have repeated numbers, except 0.')
    elif sorted(non_zero_numbers) != matched_seq:
        raise ValueError('For puzzles with X tiles, the non-zero numbers must be within [1,X], '
                          'and there will be 9-X grids labeled as 0.')
    goal_state = matched_seq
    for _ in range(9 - num_tiles):
        goal_state.append(0)
    return tuple(goal_state)

def get_manhattan_distance(from_state, to_state):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (The first one is current state, and the second one is goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    total_distance = 0
    goal_position = {value: (i // 3, i % 3) for i, value in enumerate(to_state)}
    for index, value in enumerate(from_state):
        if value != 0:
            # the position index
            current_position = (index // 3, index % 3)
            goal_position_value = goal_position[value]
            total_distance += abs(current_position[0] - goal_position_value[0]) + abs(current_position[1] - goal_position_value[1])
    return total_distance
    # return distance

def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """

    # given state, check state format and get goal_state.
    goal_state = state_check(state)
    # please remove debugging prints when you submit your code.
    # print('initial state: ', state)
    # print('goal state: ', goal_state)

    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state,goal_state)))

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

def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
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
    # return sorted(succ_states)
    return succ_unique_sorted

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

def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """

    # This is a format helper，which is only designed for format purpose.
    # define "solvable_condition" to check if the puzzle is really solvable
    # build "state_info_list", for each "state_info" in the list, it contains "current_state", "h" and "move".
    # define and compute "max_length", it might be useful in debugging
    # it can help to avoid any potential format issue.

    # given state, check state format and get goal_state.
    goal_state = state_check(state)
    # please remove debugging prints when you submit your code.
    # print('initial state: ', state)
    # print('goal state: ', goal_state)
    goal_state = list(goal_state)

    # goal_position = {value: (i // 3, i % 3) for i, value in enumerate(goal_state)}
    pq = []
    history = []
    visited = set()
    succ_cost_dict = {}
    initial_h = get_manhattan_distance(state, goal_state)
    heapq.heappush(pq, (initial_h, state, (0, initial_h, -1)))
    visited.add(tuple(state)) # reduce the complexity, ignore the visited node
    succ_cost_dict[tuple(state)] = initial_h
    history.append((state, -1)) # use history to build the tree, like the replationship between parent and children
    max_queue_length = 0
    solvable_condition = False
    # MAX_STEP = 10^
    # step = 0
    while pq:
        if len(pq) > max_queue_length:
            max_queue_length = len(pq)
            
        cost, cur_state, (g, h, parent_index) = heapq.heappop(pq) # pop the minimum cost one
        # step += 1
        # if step > MAX_STEP:
        #     print(False)
        #     return
        if cur_state == goal_state:
            index = len(history) - 1
            for idx in range(len(history)):
                if history[idx][0] == goal_state:
                    index = idx
                    break
            # path = reconstruct_path(history, len(history) - 1, state)
            path = reconstruct_path(history, index, state)
            # path.append(history[index])
            print(True)
            for i, (s, idx) in enumerate(path):
                h_value = get_manhattan_distance(s, goal_state)
                # print(f"{s} h={h_value} moves: {i}")
                print(s, "h={}".format(h_value), "moves: {}".format(i))
            # print(f"Max queue length: {max_queue_length}")
            print("Max queue length: {}".format(max_queue_length))
            # return [state for state, idx in path]
            solvable_condition = True
            break

        succ_states = get_succ(cur_state)
        # push every successors into the heap
        for succ in succ_states:
            # if tuple(succ) not in visited:
            #     new_g = g + 1
            #     new_h = get_manhattan_distance(succ, goal_state)
            #     new_cost = new_g + new_h
            #     heapq.heappush(pq, (new_cost, succ, (new_g, new_h, len(history))))
            #     history.append((succ, parent_index))
            #     visited.add(tuple(succ))
            new_g = g + 1
            new_h = get_manhattan_distance(succ, goal_state)
            new_cost = new_g + new_h
            if tuple(succ) not in visited:
                heapq.heappush(pq, (new_cost, succ, (new_g, new_h, len(history))))
                history.append((succ, parent_index))
                visited.add(tuple(succ))
                succ_cost_dict[tuple(succ)] = new_cost
            else:
                if new_cost < succ_cost_dict[tuple(succ)]:
                    heapq.heappush(pq, (new_cost, succ, (new_g, new_h, len(history))))
                    history.append((succ, parent_index))
                    visited.add(tuple(succ))
                    succ_cost_dict[tuple(succ)] = new_cost
                else:
                    continue
            
            
    if not solvable_condition:
        print(False)
        return



    # if not solvable_condition:
    #     print(False)
    #     return
    # else:
    #     print(True)

    # for state_info in state_info_list:
    #     current_state = state_info[0]
    #     h = state_info[1]
    #     move = state_info[2]
    #     print(current_state, "h={}".format(h), "moves: {}".format(move))
    # print("Max queue length: {}".format(max_length))

if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    solve([5,4,3,2,1,0,0,7,6])
    print()
