import chess
import random
from math import inf
from IPython.display import display, clear_output

def get_minimax_move(b: chess.Board, evaluation, player: bool, ply: int):
    """
    This function chooses the best move for the given board position, evaluation function, player, and ply.
    
    Parameters:
    - board: the chess board
    - evaluation: a function that returns a score given a chess board and a player
    - player: the colour of the active player (True -> white, False -> black)
    - ply: the number of move pairs (1 ply = both max AND min) that the algorithmn should look ahead.
    
    Returns:
    A single chess.Move type object.
    """
    val, move = value(b, evaluation, player, ply, float('-inf'), float('inf'))
    return move

'''
Computes the optimal minimax move for the max number of plys specified
Returns the minimax move
'''
def value(board: chess.Board, evaluation, player: bool, ply: int, alpha, beta) -> tuple[float, chess.Move | None] :
    if ply <= 0:
        eval_res = evaluation(board, player)
        return (eval_res, None)
    # not sure how to handle this since a bool is representing the color of the player and not necessarily the min and max agent
    elif player: # next agent is max agent (player)
        return max_val(board, evaluation, player, ply, alpha, beta)
    elif not player: # next agent is the min agent
        return min_value(board, evaluation, player, ply, alpha, beta)

'''
Determine the max value of the node
'''
def max_val(board: chess.Board, evaluation, player: bool, ply: int, alpha, beta) -> tuple[float, chess.Move]:
    max_val = float('-inf')
    max_move = None
    sorted_moves = order_moves(board)
    for move in sorted_moves:
        board.push(move)
        node_val, node_move = value(board, evaluation, not player, ply, alpha, beta)
        board.pop()
        if node_val > max_val:
            max_val = node_val
            max_move = move
        
        if max_val > alpha:
            alpha = max_val
        
        if alpha >= beta:
            break
    return (max_val, max_move)

'''
Determine the min value of the node
'''
def min_value(board: chess.Board, evaluation, player: bool, ply: int, alpha, beta) -> tuple[float, chess.Move]:
    min_val = float('inf')
    min_move = None
    for move in board.legal_moves:
        board.push(move)
        node_val, node_move = value(board, evaluation, not player, ply - 1, alpha, beta)
        board.pop()
        if node_val < min_val:
            min_val = node_val
            min_move = move
        
        if min_val < beta:
            beta = min_val
        
        if alpha >= beta:
            break
    return (min_val, min_move)

'''
Order the moves in terms of 
* Puts the the curr player in checkmate
* Captures a piece
* all other moves
* gives the other player a checkmate,
'''
def order_moves(board: chess.Board):
    check_moves = []
    capture_moves = []
    other_moves = []
    for move in board.legal_moves:
        if board.gives_check(move):
            check_moves.append(move)
        elif board.is_capture(move):
            capture_moves.append(move)
        else:
            other_moves.append(move)
    return [*check_moves, *capture_moves, *other_moves]



def get_position_score(board: chess.Board, player: bool):   
    """
    This function returns a score for a board position, from the given player's point of view.
    A higher score should be more favourable than a low score.
    
    Parameters:
    - board: the chess board that the knight is moving upon
    - player: the colour of the active player (True -> white, False -> black)
    
    Returns:
    A numerical value representing the score of the board position from player's point of view.
    """
    if board.is_checkmate():
        if player:
            return float('inf')
        return float('-inf')
    
    pos_val = 0
    for sq, piece in board.piece_map().items():
        pos_val += get_piece_value(player, piece)
        defended = board.is_attacked_by(player, sq)
        if board.is_attacked_by(not player, sq) and not defended:
            if is_well_defended(board, piece, player, sq):
                pos_val += 5
            else:
                pos_val -= 9
        elif board.is_attacked_by(not player, sq) and defended:
            pos_val += 2
        elif board.is_attacked_by(player, sq) and not board.is_attacked_by(not player, sq):
            pos_val += 8
            # attacking_factor = abs(get_piece_value(player, piece))*2.5
            # else:
            #     pos_val -= 9
        # player is attacking the opposite side & has good attacks factor that in
        # elif piece.color != player and board.is_attacked_by(player, sq) and not is_well_defended(board, piece, not player, sq):
        #     pos_val +=  8

        # if board.is_pinned(piece.color, sq):
        #     if piece.color == player:
        #         pos_val += 5
        #     else:
        #         pos_val -= 5
        
    return pos_val

def get_piece_value(player, piece: chess.Piece) -> int:
    pieces_val = {
        chess.KING: 1000, # need to have a king to be playing so don't fact this into score
        chess.QUEEN: 90,
        chess.ROOK: 50,
        chess.KNIGHT: 30,
        chess.BISHOP: 30,
        chess.PAWN: 10, 
    }
    val = pieces_val[piece.piece_type]
    if piece.color != player:
        val *= -1
    
    return val

def is_well_defended(board: chess.Board, piece: chess.Piece, player_color: chess.Color, square: chess.Square) -> bool:
    # other pieces of the same color defending the given piece
    defenders = board.attackers(piece.color, square)
    attackers = board.attackers(not piece.color, square)
    # print(f'Attacked by: {len(attackers)} Defended by: {len(defenders)}')
    return len(defenders) - len(attackers) >= 1

# testing q5
# engine = chess.engine.SimpleEngine.popen_uci(r"C:\Users\honey\Desktop\School\Code\cse240_assignments\Assignment2\stockfish-windows-x86-64-avx2.exe") 
"""
import chess.engine
# As above, so below; feel free to use this code to test your solution to the puzzle above.
puzzle3_board = chess.Board("1k6/2b2p2/2p1p3/1pP2p2/1P1P1P2/8/2N1P3/1K6")
engine = chess.engine.SimpleEngine.popen_uci(r"stockfish-windows-x86-64-avx2.exe")
pre_move_info = engine.analyse(puzzle3_board, chess.engine.Limit(time=0.1))

move = get_minimax_move(puzzle3_board, get_position_score, True, 1)
print(f"Move: {move} Expected: e2e3")
puzzle3_board.push(move)

post_move_info = engine.analyse(puzzle3_board, chess.engine.Limit(time=0.1))
engine.quit()
print(f"Pre-Move Score: {pre_move_info['score'].white()}")
print(f"Post-Move Score: {post_move_info['score'].white()}") 
"""
import utils
# case_name, case_board, case_depth, case_ans, case_pts = ("Puzzle 1 Visible", chess.Board("1rr5/p1p2Rpp/2Qpk3/4n1q1/4P3/8/PPP3PP/R6K"), 1, "c6d5", 5)
# student_ans = utils.testAdvSearch(get_minimax_move, get_position_score, case_board, case_depth)
# print(f'Q3: Student move: {student_ans} \tExpected Move: {case_ans}')

case_name, case_board, case_depth, case_ans, case_pts = ("Puzzle 2 Visible", chess.Board("8/1r3k2/2r1ppp1/8/5PB1/4P3/4PK2/5R2"), 3, "g4f3", 10)
student_ans = utils.testAdvSearch(get_minimax_move, get_position_score, case_board, case_depth)
print(f'Q4: Student move: {student_ans} \tExpected Move: {case_ans}')

case_name, case_board, case_depth, case_ans, case_pts = ("Puzzle 3 Visible", chess.Board("1k6/2b2p2/2p1p3/1pP2p2/1P1P1P2/8/2N1P3/1K6"), 1, "e2e3", 10)
student_ans = utils.testAdvSearch(get_minimax_move, get_position_score, case_board, case_depth)
print(f'Q5: Student move: {student_ans} \tExpected Move: {case_ans}')
