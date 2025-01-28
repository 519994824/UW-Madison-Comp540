import random
from typing import List
import time

# def time_it(func):
#     """A decorator to monitor the execution time of a function."""
#     def wrapper(*args, **kwargs):
#         start_time = time.time()  # 记录开始时间
#         result = func(*args, **kwargs)  # 执行函数
#         end_time = time.time()  # 记录结束时间
#         execution_time = end_time - start_time  # 计算执行时间
#         print(f"{func.__name__} execution time: {execution_time:.4f} seconds")
#         return result
#     return wrapper

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]
        self.depth_limit = 3

    def run_challenge_test(self):
        # Set to True if you would like to run gradescope against the challenge AI!
        # Leave as False if you would like to run the gradescope tests faster for debugging.
        # You can still get full credit with this set to False
        # return False
        return True
    
    def count_center_control(self, state, piece):
        # 定义中心位置的坐标
        center_positions = [(1, 1), (1, 2), (1, 3),
                            (2, 1), (2, 2), (2, 3),
                            (3, 1), (3, 2), (3, 3)]
        count = 0
        for row, col in center_positions:
            if state[row][col] == piece:
                count += 1
        return count
    
    def count_two_in_row(self, state, piece):
        count = 0
        # 检查水平方向的两连
        for row in state:
            for i in range(4):
                line = row[i:i+2]
                if line.count(piece) == 2:
                    count += 1
        # 检查垂直方向的两连
        for col in range(5):
            for i in range(4):
                line = [state[i][col], state[i+1][col]]
                if line.count(piece) == 2:
                    count += 1
        # 检查 '\' 方向的两连
        for i in range(4):
            for j in range(4):
                line = [state[i][j], state[i+1][j+1]]
                if line.count(piece) == 2:
                    count += 1
        # 检查 '/' 方向的两连
        for i in range(1, 5):
            for j in range(4):
                line = [state[i][j], state[i-1][j+1]]
                if line.count(piece) == 2:
                    count += 1
        return count
    
    def count_three_in_row(self, state, piece):
        count = 0
        # 检查水平方向的三连
        for row in state:
            for i in range(3):
                line = row[i:i+3]
                if line.count(piece) == 3:
                    count += 1
        # 检查垂直方向的三连
        for col in range(5):
            for i in range(3):
                line = [state[i][col], state[i+1][col], state[i+2][col]]
                if line.count(piece) == 3:
                    count += 1
        # 检查 '\' 方向的三连
        for i in range(3):
            for j in range(3):
                line = [state[i][j], state[i+1][j+1], state[i+2][j+2]]
                if line.count(piece) == 3:
                    count += 1
        # 检查 '/' 方向的三连
        for i in range(2, 5):
            for j in range(3):
                line = [state[i][j], state[i-1][j+1], state[i-2][j+2]]
                if line.count(piece) == 3:
                    count += 1
        return count
    
    def calculate_mobility(self, state, piece):
        mobility = 0
        for i in range(5):
            for j in range(5):
                if state[i][j] == piece:
                    # 检查所有相邻位置
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            x, y = i + dx, j + dy
                            if 0 <= x < 5 and 0 <= y < 5:
                                if state[x][y] == ' ':
                                    mobility += 1
        return mobility
    
    def count_potential_squares(self, state, piece):
        count = 0
        for i in range(4):  # 行索引从 0 到 3
            for j in range(4):  # 列索引从 0 到 3
                square = [state[i][j], state[i][j+1],
                        state[i+1][j], state[i+1][j+1]]
                piece_count = square.count(piece)
                empty_count = square.count(' ')
                # if piece_count == 2 and empty_count == 2:
                #     # 两个指定的棋子，两个空位
                #     count += 1
                # elif piece_count == 3 and empty_count == 1:
                #     # 三个指定的棋子，一个空位
                #     count += 2  # 给三子方块更高的权重
                if piece_count == 3 and empty_count == 1:
                    count += 1
        return count
    
    def heuristic_game_value(self, state):
        # 检查终止状态
        game_val = self.game_value(state)
        if game_val != 0:
            return game_val

        score = 0.0

        # 定义权重
        weights = {
            'my_two_in_row': 0.2,
            'my_three_in_row': 0.5,
            'opp_two_in_row': -0.3,
            'opp_three_in_row': -0.6,
            'center_control': 0.1,
            'opp_center_control': -0.15,
            # 'mobility': 0.02,
            # 'opp_mobility': -0.01,
            'my_square': 0.2,
            'opp_square': -0.3,
            # 添加更多因素...
        }

        # 计算我方棋子的优势
        score += weights['center_control'] * self.count_center_control(state, self.my_piece)
        score += weights['my_two_in_row'] * self.count_two_in_row(state, self.my_piece)
        score += weights['my_three_in_row'] * self.count_three_in_row(state, self.my_piece)
        score += weights['my_square'] * self.count_potential_squares(state, self.my_piece)
        # score += weights['mobility'] * self.calculate_mobility(state, self.my_piece)

        # 计算对手的威胁
        score += weights['opp_center_control'] * self.count_center_control(state, self.opp)
        score += weights['opp_two_in_row'] * self.count_two_in_row(state, self.opp)
        score += weights['opp_three_in_row'] * self.count_three_in_row(state, self.opp)
        score += weights['opp_square'] * self.count_potential_squares(state, self.my_piece)
        # score += weights['opp_mobility'] * self.calculate_mobility(state, self.my_piece)

        # 确保评估值在 -10 到 10 之间
        score = max(min(score, 1), -1)

        return score
    
    def is_drop_phase(self, state):
        count = len([1 for rows in state for value in rows if value != " "])
        if count == 8:
            drop_phase = False
        else:
            drop_phase = True
        return drop_phase

    def succ(self, state: List[List], piece: str) -> List[List]:
        successors = []
        drop_phase = self.is_drop_phase(state)

        if drop_phase:
            # 落子阶段
            for i in range(5):
                for j in range(5):
                    if state[i][j] == ' ':
                        new_state = [row[:] for row in state]
                        new_state[i][j] = piece
                        successors.append(new_state)
        else:
            # 移动阶段
            for i in range(5):
                for j in range(5):
                    if state[i][j] == piece:
                        # 遍历相邻位置
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                if dx == 0 and dy == 0:
                                    continue
                                x, y = i + dx, j + dy
                                if 0 <= x < 5 and 0 <= y < 5 and state[x][y] == ' ':
                                    new_state = [row[:] for row in state]
                                    new_state[i][j] = ' '
                                    new_state[x][y] = piece
                                    successors.append(new_state)
        return successors
    
    # def max_value(self, state, alpha, beta, depth):
    #     if depth >= self.depth_limit or self.game_value(state) != 0:
    #         return self.heuristic_game_value(state)
    #     value = float('-inf')
    #     for s in self.succ(state, self.my_piece):
    #         value = max(value, self.min_value(s, alpha, beta, depth + 1))
    #         if value >= beta:
    #             return value  # Beta 剪枝
    #         alpha = max(alpha, value)
    #     return value

    # def min_value(self, state, alpha, beta, depth):
    #     if depth >= self.depth_limit or self.game_value(state) != 0:
    #         return self.heuristic_game_value(state)
    #     value = float('inf')
    #     for s in self.succ(state, self.opp):
    #         value = min(value, self.max_value(s, alpha, beta, depth + 1))
    #         if value <= alpha:
    #             return value  # Alpha 剪枝
    #         beta = min(beta, value)
    #     return value

    def max_value(self, state, alpha, beta, depth):
        if self.game_value(state) != 0:
            return self.game_value(state)

        if depth >= self.depth_limit:
            return self.heuristic_game_value(state)

        for successor in self.succ(state, self.my_piece):
            alpha = max(alpha, self.min_value(successor, alpha, beta, depth + 1))
            if alpha >= beta:
                return beta
        return alpha

    def min_value(self, state, alpha, beta, depth):
        if self.game_value(state) != 0:
            return self.game_value(state)

        if depth >= self.depth_limit:
            return self.heuristic_game_value(state)

        for successor in self.succ(state, self.opp):
            beta = min(beta, self.max_value(successor, alpha, beta, depth + 1))
            if alpha >= beta:
                return beta
        return beta
    
    # @time_it 
    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
        drop_phase = self.is_drop_phase(state)
        best_value = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')

        if drop_phase:
            # 生成所有可能的落子走法
            for i in range(5):
                for j in range(5):
                    if state[i][j] == ' ':
                        move = [(i, j)]
                        new_state = [row[:] for row in state]
                        new_state[i][j] = self.my_piece
                        value = self.min_value(new_state, alpha, beta, 1)
                        if value > best_value:
                            best_value = value
                            best_move = move
                        alpha = max(alpha, best_value)
        else:
            # 生成所有可能的移动走法
            for i in range(5):
                for j in range(5):
                    if state[i][j] == self.my_piece:
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                if dx == 0 and dy == 0:
                                    continue
                                x, y = i + dx, j + dy
                                if 0 <= x < 5 and 0 <= y < 5 and state[x][y] == ' ':
                                    move = [(x, y), (i, j)]
                                    new_state = [row[:] for row in state]
                                    new_state[i][j] = ' '
                                    new_state[x][y] = self.my_piece
                                    value = self.min_value(new_state, alpha, beta, 1)
                                    if value > best_value:
                                        best_value = value
                                        best_move = move
                                    alpha = max(alpha, best_value)
        return best_move
    
    import time



    def move_to_detection(self, state, move_to):
        if 0 <= move_to[0] < 5 and 0 <= move_to[1] < 5 and state[move_to[0]][move_to[1]] == ' ':
            return True
        return False
        
    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and box wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # TODO: check \ diagonal wins
        for col in range(2):
            for row in range(2):
                if state[row][col] != ' ' and state[row][col] == state[row+1][col+1] == state[row+2][col+2] == state[row+3][col+3]:
                    return 1 if state[i][col]==self.my_piece else -1
        # TODO: check / diagonal wins
        for col in range(4,2,-1):
            for row in range(2):
                if state[row][col] != ' ' and state[row][col] == state[row+1][col-1] == state[row+2][col-2] == state[row+3][col-3]:
                    return 1 if state[i][col]==self.my_piece else -1
        # TODO: check box wins
        for col in range(4):
            for row in range(4):
                if state[row][col] != ' ' and state[row][col] == state[row+1][col] == state[row][col+1] == state[row+1][col+1]:
                    return 1 if state[i][col]==self.my_piece else -1

        return 0 # no winner yet

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
