import chess
from .interface import Interface

# Piece values for evaluation
BASE_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,

    chess.BISHOP: 3,
    chess.ROOK: 5,

    chess.QUEEN: 9,
    chess.KING: 0
}

POSITION_TABLES = {
    chess.PAWN: [
        # 8x8 grid with position values
        # Center pawns are worth more
        # Advancing pawns worth progressively more
    ]
}



PIECE_SQUARE_TABLES = {
    'middlegame': {
        chess.PAWN: [
            [100, 100, 100, 100, 100, 100, 100, 100], # Rank 8 (promotion rank - huge bonus)
            [ 50,  50,  50,  50,  50,  50,  50,  50], # Rank 7 (near promotion)
            [ 10,  10,  20,  30,  30,  20,  10,  10], # Rank 6
            [  5,   5,  10,  25,  25,  10,   5,   5], # Rank 5
            [  0,   0,   0,  20,  20,   0,   0,   0], # Rank 4
            [  5,  -5, -10,   0,   0, -10,  -5,   5], # Rank 3
            [  5,  10,  10, -20, -20,  10,  10,   5], # Rank 2
            [  0,   0,   0,   0,   0,   0,   0,   0]  # Rank 1
        ],
        chess.KNIGHT: [
            [-50, -40, -30, -30, -30, -30, -40, -50], # Back rank (bad)
            [-40, -20,   0,   5,   5,   0, -20, -40],
            [-30,   5,  10,  15,  15,  10,   5, -30],
            [-30,   0,  15,  20,  20,  15,   0, -30],
            [-30,   5,  15,  20,  20,  15,   5, -30],
            [-30,   0,  10,  15,  15,  10,   0, -30],
            [-40, -20,   0,   0,   0,   0, -20, -40],
            [-50, -40, -30, -30, -30, -30, -40, -50]
        ],
        chess.BISHOP: [
            [-20, -10, -10, -10, -10, -10, -10, -20], # Back rank
            [-10,   5,   0,   0,   0,   0,   5, -10],
            [-10,  10,  15,  10,  10,  15,  10, -10],
            [-10,   0,  10,  15,  15,  10,   0, -10],
            [-10,   5,  15,  20,  20,  15,   5, -10], # Strong central diagonal control
            [-10,  10,   5,  10,  10,   5,  10, -10],
            [-10,   5,   0,   0,   0,   0,   5, -10],
            [-20, -10, -10, -10, -10, -10, -10, -20]
        ],
        chess.ROOK: [
            [ 15,  15,  15,  20,  20,  15,  15,  15], # Strong on 8th rank
            [ 15,  20,  20,  20,  20,  20,  20,  15], # Strong on 7th rank
            [  0,   5,   5,  10,  10,   5,   5,   0],
            [  0,   0,   5,  10,  10,   5,   0,   0],
            [  0,   0,   5,  10,  10,   5,   0,   0],
            [  0,   0,   5,   5,   5,   5,   0,   0],
            [  5,   5,  10,  10,  10,  10,   5,   5],
            [  0,   0,   5,   5,   5,   5,   0,   0]
        ],
        chess.QUEEN: [
            [-20, -10, -10,  -5,  -5, -10, -10, -20],
            [-10,   0,   5,   0,   0,   0,   0, -10],
            [-10,   5,   5,   5,   5,   5,   0, -10],
            [  0,   0,   5,   5,   5,   5,   0,  -5],
            [ -5,   0,   5,   5,   5,   5,   0,  -5],
            [-10,   0,   5,   5,   5,   5,   0, -10],
            [-10,   0,   0,   0,   0,   0,   0, -10],
            [-20, -10, -10,  -5,  -5, -10, -10, -20]
        ],
        chess.KING: [
            [ 30,  20,   0, -10, -10,   0,  20,  30], # Corners safer
            [ 20,  10,  -5, -10, -10,  -5,  10,  20],
            [-10, -20, -20, -20, -20, -20, -20, -10],
            [-20, -30, -30, -40, -40, -30, -30, -20],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [ 20,  10, -20, -30, -30, -20,  10,  20]  # Castle positions better
        ]
    },
    
    'endgame': {
        # Most pieces keep their middlegame tables except the king
        chess.KING: [
            [-20, -10, -10, -10, -10, -10, -10, -20],
            [-10,   0,  10,  10,  10,  10,   0, -10],
            [-10,  10,  20,  30,  30,  20,  10, -10],
            [-10,  10,  30,  40,  40,  30,  10, -10],
            [-10,  10,  30,  40,  40,  30,  10, -10],
            [-10,  10,  20,  30,  30,  20,  10, -10],
            [-10,   0,  10,  10,  10,  10,   0, -10],
            [-20, -10, -10, -10, -10, -10, -10, -20]
        ]
    }
}

#Here, we will add weights based on how much control a piece has, what type of safety each piece is in, etc



# center control 
def count_center_control(board, color, CENTER_UNIT_VALUE):
    #counts how many squares are under attack in the center
    middle_squares = [chess.C3, chess.C4, chess.C5, chess.C6, chess.D3, chess.D4, chess.D5, chess.D6, chess.E3, chess.E4, chess.E5, chess.E6, chess.F3, chess.F4, chess.F5]
    control_count = 0
    for square in middle_squares:
        if board.is_attacked_by(color, square):
            control_count += 1

    return control_count


def is_defended(board, square, color, piece_values=BASE_PIECE_VALUES):
    defenders = 0
    attackers = 0
    value = 0
    for position in board.attackers(color, square):
        piece = board.piece_at(position)
        value = piece_values[piece.piece_type]
        defenders += value
    for position in board.attackers(not color, square):
        piece = board.piece_at(position)
        value = piece_values[piece.piece_type]
        attackers += value
    if defenders > attackers: 
        return True
    return False  

def capture(board, color):
    for square in chess.SQUARES():
        if board.attackers(square, )

def bishop_pos_weight(BISHOP_W, board, color, square):
    bishop_position_points = BISHOP_W
    if is_defended(board, square, color):
        bishop_position_points += 0.02
    bishop_position_points += 0.02 * len(board.attacks(square))

    return bishop_position_points

def knight_pos_weight(KNIGHT_W, board, color, square):
    knight_position_points = KNIGHT_W
    if is_defended(board, square, color):
        knight_position_points += 0.02
    knight_position_points += 0.02 * len(board.attacks(square))

    return knight_position_points

def LINKED_POINTS(board, color, LINKED_BONUS):
    for square in chess.SQUARES:
        if board.piece_at_square() == chess.PAWN:
            if board.is_attacked_by(chess.WHITE, square):
                attackers = board.attackers(chess.WHITE, square)
                for piece in attackers:
                    if piece == chess.PAWN and piece.color == chess.WHITE:
                        LINKED_BONUS += 0.01
    return LINKED_BONUS
                
            
def pawn_metrics(board, color, PASSED_PAWN_VALUE):
    white_pawn_metrics = LINKED_POINTS(chess.board, chess.WHITE)
    black_pawn_metrics = LINKED_POINTS(chess.board, chess.BLACK)
#sums all of the pawn features with penalties and values
    pawn_structure_score = (white_pawn_metrics.passed * PASSED_PAWN_VALUE - white_pawn_metrics.doubled * DOUBLED_PAWN_PENALTY - white_pawn_metrics.isolated * ISOLATED_PAWN_PENALTY)
    pawn_structure_score -= (black_pawn_metrics.passed * PASSED_PAWN_VALUE - black_pawn_metrics.doubled * DOUBLED_PAWN_PENALTY - black_pawn_metrics.isolated * ISOLATED_PAWN_PENALTY)

    return pawn_structure_score
#finds the sum of white's pawn structure based off of the 3 major pawn weights    
     
     


#need to make white safety score
#need to make black safety score

def king_safety(board, color):
    # 1. Pawn shield
    # 2. Enemy attacks around king
    # 3. Castling status (position safety)

    safety_score = 0.0

    #find the king
    king_square = board.king(color)
    if king_square is None:
        return 0
    
    #finds the coordinates
    king_file = chess.square_file(king_square)
    king_rank = chess.square_file(king_square)

    #1. pawn shield
    for file_offset in [-1, 0, 1]:
        check_file = king_file + file_offset
        if 0 <= check_file <= 7:
            #this checks a square ahead of the king
            if color == chess.WHITE:
                check_rank = king_rank + 1
                if check_rank <= 7:
                    check_square = chess.square(check_file, check_rank)
                    piece = board.piece_at(check_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        safety_score += 0.15
            else:
                check_rank = king_rank - 1
                if check_rank >= 7:
                    check_square = chess.square(check_file, check_rank)
                    piece = board.piece_at(check_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        safety_score += 0.15

    #2. ENEMY attacks: penalizes the squares around the king when it's under attack
    enemy_color = not color
    for file_offset in [-1, 0, 1]:
        for rank_offset in [-1, 0, 1]:
            check_file = king_file + file_offset
            check_rank = king_rank + rank_offset

            if 0 <= check_file <= 7 and 0 <= check_rank <= 7:
                check_square - chess.square(check_file, check_rank)
                if board.is_attacked_by(enemy_color, check_square):
                    safety_score -= 0.08


    #3 castling bonus: rewards when the king is castled 
    if color == chess.WHITE:
        if king_square in [chess.G1, chess.C1]:
            safety_score += 0.25
    else: 
        if king_square in [chess.G8, chess.C8]:
            safety_score += 0.25


    return safety_score


def get_game_phase(board):
    """
    Determines if we're in middlegame or endgame.
    Returns a value between 0 (pure endgame) and 1 (pure middlegame)
    """
    # Material weights for phase calculation
    PHASE_WEIGHTS = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 1,
        chess.ROOK: 2,
        chess.QUEEN: 4,
        chess.KING: 0
    }
    
     # Maximum phase score is when all pieces are on board
    # 2 knights (2), 2 bishops (2), 2 rooks (4), 1 queen (4) per side = 24 total
    
    MIDDLEGAME_THRESHOLD = 20
    ENDGAME_THRESHOLD = 4
    
    # Current phase score
    phase_weight = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type != chess.KING:
            phase+= PHASE_WEIGHTS[piece.piece_type]

    # Convert to a 0-1 scale where:
    # 1.0 = pure middlegame (all pieces present)
    # 0.0 = pure endgame (only kings and pawns)
    return min(1.0, phase / MAX_PHASE)


#positional knowledge, add more knowledge based on where they are in the board
#opening, middle (10 - 15 moves, most pieces are developed), endgame (less than 13 points of material each)
#Take time during meetings so that the piece algorithms don't conflict with eachother

def evaluate_board(board):
    
    """
    Evaluate the board based on material count.
    Returns score from WHITE's perspective.
    Positive score favors white, negative favors black.
    """
    
    color = chess.WHITE if board.turn == chess.WHITE else chess.BLACK

    #we declare our weights here in order to calculate how optimal each side's position is
    W_positional = 0.3
    W_mobility = 0.2
    W_center_control = 0.2
    W_pawn_structure = 0.1
    W_king_safety = 0.2

    #positional weights by piece
    BISHOP_W = 0.25
    ROOK_W = 0.25
    KNIGHT_W = 0.25
    QUEEN_W = 0.25

    #pawn weights
    CENTER_UNIT_VALUE = 0.05 #per square value for each controlled square
    PASSED_PAWN_VALUE = 0.5 #this is the bonus for having a passed pawn (cannot be stopped from promoting)
    DOUBLED_PAWN_PENALTY = 0.2 #penalty for having 2 pawns on the same file (cannot defend eachother )
    ISOLATED_PAWN_PENALTY = 0.15 #penalty for pawn with no friendly pawns on adjacent files
    LINKED_BONUS = 0.1
    
    Total_W = W_positional + W_mobility + W_center_control + W_pawn_structure + W_king_safety + CENTER_UNIT_VALUE + PASSED_PAWN_VALUE - DOUBLED_PAWN_PENALTY - ISOLATED_PAWN_PENALTY


    white_pawn_structure_score = pawn_metrics(chess.board, chess.WHITE, PASSED_PAWN_VALUE)
    black_pawn_structure_score = pawn_metrics(chess.board, chess.BLACK, PASSED_PAWN_VALUE)

    white_king_safety_score = king_safety(chess.board, chess.WHITE)
    black_king_safety_score = king_safety(chess.board, chess.BLACK)

    white_center = count_center_control(chess.board, chess.WHITE, CENTER_UNIT_VALUE)
    black_center = count_center_control(chess.board, chess.BLACK, CENTER_UNIT_VALUE)

    center_control_score = (white_center - black_center) * CENTER_UNIT_VALUE * W_center_control

    score += center_control_score

    if board.can_claim_threefold_repetition():
        if board.turn == chess.WHITE:
            score = float("-inf")
            return score
        elif board.turn == chess.BLACK:
            score = float("inf")
            return score
            
    if board.is_checkmate():
        # If it's White's turn and checkmate, White lost (bad for White)
        # If it's Black's turn and checkmate, Black lost (good for White)
        return float('-inf') if board.turn == chess.WHITE else float('inf')
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = BASE_PIECE_VALUES[piece.piece_type]
            score += value if piece.color == chess.WHITE else -value

    white_king_safety = white_safety_score(chess.board, chess.WHITE)
    black_king_safety = black_safety_score(chess.board, chess.BLACK)
    king_safety_score_total = (white_king_safety - black_king_safety)

    score += king_safety_score_total
            
    score = score * Total_W

    return score

def minimax(board, depth, alpha, beta, maximizing_player):
    """
    Minimax algorithm with alpha-beta pruning.
    Always evaluates from White's perspective.
    
    Args:
        board: chess.Board object
        depth: remaining depth to search
        alpha: best value for maximizer
        beta: best value for minimizer
        maximizing_player: True if White's turn, False if Black's turn
    
    Returns:
        Best evaluation score from White's perspective
    """
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)
    
    if maximizing_player:
        max_eval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval

def find_best_move(board, depth):
    """
    Find the best move for the current player.
    
    Args:
        board: chess.Board object
        depth: search depth
    
    Returns:
        Best move in UCI notation (e.g., 'e2e4')
    """
    best_move = None
    alpha = float('-inf')
    beta = float('inf')
    
    if board.turn == chess.WHITE:
        # White wants to MAXIMIZE the score
        best_value = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            board_value = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            
            if board_value > best_value:
                best_value = board_value
                best_move = move
            alpha = max(alpha, best_value)
    else:
        # Black wants to MINIMIZE the score
        best_value = float('inf')
        for move in board.legal_moves:
            board.push(move)
            board_value = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            
            if board_value < best_value:
                best_value = board_value
                best_move = move
            beta = min(beta, best_value)
    
    return best_move

def play(interface: Interface, color = "w"):
    search_depth = 4  # Can be any positive number
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board = chess.Board(fen)

    # color = interface.input()

    if color == "b":
        move = interface.input()
        board.push_san(move)

    while True:
        best_move = find_best_move(board, search_depth)
        interface.output(board.san(best_move))
        board.push(best_move)

        move = interface.input()
        board.push_san(move)
        # print(board)
