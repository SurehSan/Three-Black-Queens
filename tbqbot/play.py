import chess
import chess.polyglot 
from .interface import Interface

'''
Overview of the functions of the program:
-Transposition table w/ storing and probing functionalities
-Piece square tables
-evalutation helpers (king safety, pawn structure, mobility, positional, and center control)
-Mobility is dynamic, how many legal moves are possible for each square
-Positional is static, where the pieces are and how good they are 
-Get game phase, determines whether it is opening, middle game, or end game 
-Phase weights (helps determine phase based on importance of each piece)
-Order moves (move ordering for optimization)
-Is defended (determines if a piece is defended)
-Is outposted (determines if a pawn cannot be attacked)
-Minimax algorithm 
'''

# Transposition table for caching evaluated positions
transposition_table = {}
MAX_TT_SIZE = 1000000  # Limit cache size
#stores previously evaluated chess positions so you don't have to calculate them again

#Store position evaluation in transposition table
def store_tt(board_hash, depth, score, flag):
    if len(transposition_table) > MAX_TT_SIZE:
        transposition_table.clear()  #Clears when too large
    transposition_table[board_hash] = (depth, score, flag)


#Check if position was already evaluated
def probe_tt(board_hash, depth, alpha, beta):
    if board_hash in transposition_table:
        tt_depth, tt_score, tt_flag = transposition_table[board_hash]
        if tt_depth >= depth:
            if tt_flag == 0:  #Exact score
                return tt_score
            elif tt_flag == 1 and tt_score >= beta:  #Lower bound
                return tt_score
            elif tt_flag == 2 and tt_score <= alpha:  #Upper bound
                return tt_score
    return None

#flatten the array by combining all rows into a single list
def flatten_pst(pst_8x8):
    return [val for rank in reversed(pst_8x8) for val in rank]

#Piece values for evaluation
BASE_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}


#Piece square tables, static position bonuses/penalities for each piece type on each square
PST_PAWN = flatten_pst([
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [ 50,  50,  50,  50,  50,  50,  50,  50],
    [ 10,  10,  20,  30,  30,  20,  10,  10],
    [  5,   5,  10,  25,  25,  10,   5,   5],
    [  0,   0,   0,  20,  20,   0,   0,   0],
    [  5,  -5, -10,   0,   0, -10,  -5,   5],
    [  5,  10,  10, -20, -20,  10,  10,   5],
    [  0,   0,   0,   0,   0,   0,   0,   0]
])

PST_KNIGHT = flatten_pst([
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20,   0,   0,   0,   0, -20, -40],
    [-30,   0,  10,  15,  15,  10,   0, -30],
    [-30,   5,  15,  20,  20,  15,   5, -30],
    [-30,   0,  15,  20,  20,  15,   0, -30],
    [-30,   5,  10,  15,  15,  10,   5, -30],
    [-40, -20,   0,   5,   5,   0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50]
])

PST_BISHOP = flatten_pst([
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-10,   0,   5,  10,  10,   5,   0, -10],
    [-10,   5,   5,  10,  10,   5,   5, -10],
    [-10,   0,  10,  10,  10,  10,   0, -10],
    [-10,  10,  10,  10,  10,  10,  10, -10],
    [-10,   5,   0,   0,   0,   0,   5, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20]
])

PST_ROOK = flatten_pst([
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [  5,  10,  10,  10,  10,  10,  10,   5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [  0,   0,   0,   5,   5,   0,   0,   0]
])

PST_QUEEN = flatten_pst([
    [-20, -10, -10,  -5,  -5, -10, -10, -20],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-10,   0,   5,   5,   5,   5,   0, -10],
    [ -5,   0,   5,   5,   5,   5,   0,  -5],
    [  0,   0,   5,   5,   5,   5,   0,  -5],
    [-10,   5,   5,   5,   5,   5,   0, -10],
    [-10,   0,   5,   0,   0,   0,   0, -10],
    [-20, -10, -10,  -5,  -5, -10, -10, -20]
])

# King safety is paramount in middlegame
PST_KING_MG = flatten_pst([
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [ 20,  20,   0,   0,   0,   0,  20,  20],
    [ 20,  30,  10,   0,   0,  10,  30,  20]  # Castled positions
])

# King becomes an active piece in endgame
PST_KING_EG = flatten_pst([
    [-50, -40, -30, -20, -20, -30, -40, -50],
    [-30, -20, -10,   0,   0, -10, -20, -30],
    [-30, -10,  20,  30,  30,  20, -10, -30],
    [-30, -10,  30,  40,  40,  30, -10, -30],
    [-30, -10,  30,  40,  40,  30, -10, -30],
    [-30, -10,  20,  30,  30,  20, -10, -30],
    [-30, -30,   0,   0,   0,   0, -30, -30],
    [-50, -30, -30, -30, -30, -30, -30, -50]
    ])


#Determines if we're in middlegame or endgame.
#Returns a value between 0 (pure endgame) and 1 (pure middlegame)
#UNUSED - could be used to blend middlegame
#and endgame piece-square tables or adjust evaluation weights dynamically.
def get_game_phase(board):
    # Maximum phase score is when all pieces are on board
    # 2 knights (2), 2 bishops (2), 2 rooks (4), 1 queen (4) per side = 24 total
    MAX_PHASE = 24
    phase_score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type != chess.KING:
            phase_score += PHASE_WEIGHTS[piece.piece_type]
    # Convert to a 0-1 scale where:
    # 1.0 = pure middlegame (all pieces present)
    # 0.0 = pure endgame (only kings and pawns)
    return min(1.0, phase_score / MAX_PHASE)

#Material weights for phase calculation
PHASE_WEIGHTS = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 1,
        chess.ROOK: 2,
        chess.QUEEN: 4,
        chess.KING: 0
    }

#center control 
def count_center_control(board, color, CENTER_UNIT_VALUE):
    #counts how many squares are under attack in the center
    middle_squares = [chess.C3, chess.C4, chess.C5, chess.C6, 
                      chess.D3, chess.D4, chess.D5, chess.D6, 
                      chess.E3, chess.E4, chess.E5, chess.E6, 
                      chess.F3, chess.F4, chess.F5, chess.F6]
    control_count = 0
    for square in middle_squares:
        if board.is_attacked_by(color, square):
            control_count += 1

    #Return the count multiplied by the unit value for consistent scoring
    return control_count * CENTER_UNIT_VALUE



#Simple but effective move ordering for optimization
def order_moves(board, moves):
    captures = []
    non_captures = []
    #Checks captures first (likely to cause cutoffs)
    #Order moves to improve alpha-beta pruning
    for move in moves:
        if board.is_capture(move):
            captures.append(move)
        else:
            non_captures.append(move)
    
    #Return captures first, then other moves
    return captures + non_captures

#checks if a piece is defended
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


def is_outposted(board, color, square):
    """
    Check if a piece (typically knight) is on an outpost.
    An outpost is a square that:
    1. Is defended by a friendly pawn
    2. Cannot be attacked by enemy pawns
    """
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    
    # Check if defended by a friendly pawn on adjacent files
    has_pawn_support = False
    
    if color == chess.WHITE:
        # White pawns defend from behind (lower ranks)
        if rank > 0:
            for file_offset in [-1, 1]:
                check_file = file + file_offset
                if 0 <= check_file <= 7:
                    check_square = chess.square(check_file, rank - 1)
                    piece = board.piece_at(check_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        has_pawn_support = True
                        break
        
        # Check if enemy pawns can attack this square
        # Enemy pawns would be on adjacent files, ahead of this square
        for file_offset in [-1, 1]:
            check_file = file + file_offset
            if 0 <= check_file <= 7:
                for check_rank in range(rank, 8):
                    check_square = chess.square(check_file, check_rank)
                    piece = board.piece_at(check_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == chess.BLACK:
                        return False  # Enemy pawn can attack
    else:
        # Black pawns defend from behind (higher ranks)
        if rank < 7:
            for file_offset in [-1, 1]:
                check_file = file + file_offset
                if 0 <= check_file <= 7:
                    check_square = chess.square(check_file, rank + 1)
                    piece = board.piece_at(check_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        has_pawn_support = True
                        break
        
        # Check if enemy pawns can attack this square
        for file_offset in [-1, 1]:
            check_file = file + file_offset
            if 0 <= check_file <= 7:
                for check_rank in range(0, rank + 1):
                    check_square = chess.square(check_file, check_rank)
                    piece = board.piece_at(check_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
                        return False  # Enemy pawn can attack
    
    return has_pawn_support

def bishop_pos_weight(board, color, square):

    bishop_position_points = 0
    if is_defended(board, square, color):
        bishop_position_points += 0.02
    bishop_position_points += 0.02 * len(board.attacks(square))

    return bishop_position_points

def knight_pos_weight(board, color, square):
    knight_position_points = 0
    if is_defended(board, square, color):
        knight_position_points += 0.02
    if is_outposted(board, color, square):
        knight_position_points += 0.5
    knight_position_points += 0.02 * len(board.attacks(square))

    return knight_position_points 

def rook_pos_weight(board, color, square):
    rook_position_points = 0
    if is_defended(board, square, color):
        rook_position_points += 0.02
    rook_position_points += 0.02 * len(board.attacks(square))
    return rook_position_points

def queen_pos_weight(board, color, square):
    queen_position_points = 0
    if is_defended(board, square, color):
        queen_position_points += 0.02
    queen_position_points += 0.01 * len(board.attacks(square))
    return queen_position_points

def LINKED_POINTS(board, color, square, LINKED_BONUS):
    """ CHANGED THIS FUNCTION!!!
    Calculate bonus for linked/connected pawns"""
    piece = board.piece_at(square)
    if piece and piece.piece_type == chess.PAWN and piece.color == color:
        if board.is_attacked_by(color, square): #defeneded by our own
            attackers = board.attackers(color, square)

            for attacker_square in attackers:
                attacker_piece = board.piece_at(attacker_square)

                if attacker_piece and attacker_piece.piece_type == chess.PAWN and attacker_piece.color == color:
                    LINKED_BONUS += 0.01
    return LINKED_BONUS

                
            
def pawn_metrics(board):
    """Calculate pawn structure metrics.
    OPTIMIZED: Now only calculates for both colors once, not redundantly"""
    #only calculates linked pawns. 
    #NEED TO: Implement passed, doubled, and isolated pawn detection

    white_pawn_metrics = 0.0
    black_pawn_metrics = 0.0
    for square in chess.SQUARES:
        white_pawn_metrics = LINKED_POINTS(board, chess.WHITE, square, white_pawn_metrics)
        black_pawn_metrics = LINKED_POINTS(board, chess.BLACK, square, black_pawn_metrics)


    pawn_structure_score = white_pawn_metrics - black_pawn_metrics
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
    king_rank = chess.square_rank(king_square)

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
                if check_rank >= 0:  #FIXED!!! Black pawns need to check rank >= 0, not >= 7
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
                check_square = chess.square(check_file, check_rank)
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


#positional weights by piece
BISHOP_W = 0.35 
ROOK_W = 0.2    
KNIGHT_W = 0.25  
QUEEN_W = 0.2   

def evaluate_positional(board, square, game_phase=None):
    """
    Evaluate piece positions using piece-square tables.
    Returns score from WHITE's perspective.
    Positive score favors white, negative favors black.
    """
    
    piece = board.piece_at(square)
    if piece is None:
        return 0

    if game_phase is None:
        phase = get_game_phase(board)
    else:
        phase = game_phase
        
    piece_type = piece.piece_type
    color = piece.color
    
    # 1. Calculate extra positional points (outposts, defense, etc.)
    extra_piece_points = 0
    if piece_type == chess.BISHOP:
        extra_piece_points = bishop_pos_weight(board, color, square) * BISHOP_W
    elif piece_type == chess.KNIGHT:
        extra_piece_points = knight_pos_weight(board, color, square) * KNIGHT_W
    elif piece_type == chess.ROOK:
        extra_piece_points = rook_pos_weight(board, color, square) * ROOK_W
    elif piece_type == chess.QUEEN:
        extra_piece_points = queen_pos_weight(board, color, square) * QUEEN_W

    # 2. Get positional value from piece-square tables (PSTs)
    # Use chess.square_mirror to get the correct square for Black's perspective
    pst_square = square if color == chess.WHITE else chess.square_mirror(square)
    
    position_score_mg = 0
    position_score_eg = 0

    if piece_type == chess.PAWN:
        position_score_mg = PST_PAWN[pst_square]
        position_score_eg = PST_PAWN[pst_square]
    elif piece_type == chess.KNIGHT:
        position_score_mg = PST_KNIGHT[pst_square]
        position_score_eg = PST_KNIGHT[pst_square]
    elif piece_type == chess.BISHOP:
        position_score_mg = PST_BISHOP[pst_square]
        position_score_eg = PST_BISHOP[pst_square]
    elif piece_type == chess.ROOK:
        position_score_mg = PST_ROOK[pst_square]
        position_score_eg = PST_ROOK[pst_square]
    elif piece_type == chess.QUEEN:
        position_score_mg = PST_QUEEN[pst_square]
        position_score_eg = PST_QUEEN[pst_square]
    elif piece_type == chess.KING:
        position_score_mg = PST_KING_MG[pst_square]
        position_score_eg = PST_KING_EG[pst_square]

    # 3. Blend MG and EG Scores (Tapered Eval)
    # PST values are in centipawns, so divide by 100 to get pawn units
    pst_score = ((position_score_mg * phase) + (position_score_eg * (1.0 - phase))) / 100.0
        
    # 4. Combine scores and return
    total_score = pst_score + extra_piece_points
    
    # Return a positive score for White, negative for Black
    return total_score if color == chess.WHITE else -total_score



def evaluate_mobility(board):
    """
    Evaluate mobility by counting legal moves available to each side.
    More moves = better position (more options and control).
    Returns score from WHITE's perspective.
    """
    # Count moves for the current side
    current_side_mobility = board.legal_moves.count()
    
    # Use a null move to count opponent's mobility
    # Push a null move to switch turns
    board.push(chess.Move.null())
    opponent_mobility = board.legal_moves.count()
    board.pop()  # Undo the null move
    
    # Calculate mobility difference from White's perspective
    if board.turn == chess.WHITE:
        white_mobility = current_side_mobility
        black_mobility = opponent_mobility
    else:
        white_mobility = opponent_mobility
        black_mobility = current_side_mobility
    
    # Return difference scaled down
    return (white_mobility - black_mobility) / 10.0


#positional knowledge, add more knowledge based on where they are in the board
#opening, middle (10 - 15 moves, most pieces are developed), endgame (less than 13 points of material each)
#Take time during meetings so that the piece algorithms don't conflict with eachother
def evaluate_board(board):
    
    """
    Evaluate the board based on material count.
    Returns score from WHITE's perspective.
    Positive score favors white, negative favors black.
    """
    
    score = 0

    color = chess.WHITE if board.turn == chess.WHITE else chess.BLACK

    #we declare our weights here in order to calculate how optimal each side's position is
        # Calculate game phase once

    phase = get_game_phase(board)
    if phase == 1:
        W_positional = 0.3  
        W_mobility = 0.1    
        W_center_control = 0.2 #.2
        W_pawn_structure = 0.25 #.1
        W_king_safety = 0.15 #.2
    elif phase < 1 and phase > 0:
        W_positional = 0.2  
        W_mobility = 0.2    
        W_center_control = 0.2 #.2
        W_pawn_structure = 0.1 #.1
        W_king_safety = 0.3 #.2   
    elif phase == 0:
        W_positional = .1  
        W_mobility = .1    
        W_center_control = .1 #.2
        W_pawn_structure = .6 #.1
        W_king_safety = .1 #.2  
    

    #pawn weights
    CENTER_UNIT_VALUE = 0.05 #per square value for each controlled square
    PASSED_PAWN_VALUE = 0.5 #this is the bonus for having a passed pawn (cannot be stopped from promoting) - Not yet used
    DOUBLED_PAWN_PENALTY = 0.2 #penalty for having 2 pawns on the same file (cannot defend eachother) -  Not yet used
    ISOLATED_PAWN_PENALTY = 0.15 #penalty for pawn with no friendly pawns on adjacent files - Not yet used
    LINKED_BONUS = 0.1  # Used in LINKED_POINTS() function
    
    # OPTIMIZED: Calculate game phase once and reuse it
    
    # Evaluate mobility (number of legal moves)
    mobility_score = evaluate_mobility(board)
    score += (mobility_score * W_mobility) * 0.1

    white_king_safety_score = king_safety(board, chess.WHITE)
    black_king_safety_score = king_safety(board, chess.BLACK)

    king_safety_score_total = (white_king_safety_score - black_king_safety_score)

    score += (king_safety_score_total * W_king_safety) * 0.1

    white_center = count_center_control(board, chess.WHITE, CENTER_UNIT_VALUE)
    black_center = count_center_control(board, chess.BLACK, CENTER_UNIT_VALUE)

    # count_center_control now returns already multiplied value
    center_control_score = white_center - black_center

    score += (center_control_score * W_center_control) *0.1
            
    if board.is_checkmate():
        # If it's White's turn and checkmate, White lost 
        # If it's Black's turn and checkmate, Black lost
        return float('-inf') if board.turn == chess.WHITE else float('inf')
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    # Material count
    material_score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = BASE_PIECE_VALUES[piece.piece_type]
            material_score += value if piece.color == chess.WHITE else -value
            

            
    # Call pawn_metrics once, it calculates both sides
    pawn_structure_score_total = pawn_metrics(board)
    score += (pawn_structure_score_total * W_pawn_structure) * 0.1
    
    # Evaluate positional scores for all pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Evaluate positional score (piece-square tables)
            positional_score = evaluate_positional(board, square, phase)
            score += (positional_score * W_positional) * 0.1
    
    #combine all evaluation components
    total_score = material_score + score  # score already includes king_safety and center_control weighted
    
    return total_score

def minimax(board, depth, alpha, beta, maximizing_player):
    """
    Minimax algorithm with alpha-beta pruning.
    Always evaluates from White's perspective.
    
    Args:
        board: board object
        depth: remaining depth to search
        alpha: best value for maximizer
        beta: best value for minimizer
        maximizing_player: True if White's turn, False if Black's turn
    
    Returns:
        Best evaluation score from White's perspective
    """

    board_hash = chess.polyglot.zobrist_hash(board)
    tt_result = probe_tt(board_hash, depth, alpha, beta)
    if tt_result is not None:
        return tt_result

    if depth == 0 or board.is_game_over():
        score = evaluate_board(board)
        store_tt(board_hash, depth, score, 0)
        return score
    
    #Order moves (captures first) for better pruning
    ordered_moves = order_moves(board, board.legal_moves)
    
    if maximizing_player:
        max_eval = float('-inf')
        for move in ordered_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Beta cutoff

        flag = 1 if beta <= alpha else 0
        store_tt(board_hash, depth, max_eval, flag)
        return max_eval
    else:
        min_eval = float('inf')
        for move in ordered_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Alpha cutoff
       
        flag = 2 if beta <= alpha else 0
        store_tt(board_hash, depth, min_eval, flag)
        return min_eval
    

def find_best_move(board, depth):
    """
    Find the best move for the current player.
    
    Args:
        board: board object
        depth: search depth
    
    Returns:
        Best move in UCI notation (e.g., 'e2e4')
    """
    best_move = None
    alpha = float('-inf')
    beta = float('inf')
    
    if board.turn == chess.WHITE:
        # White wants to MAXIMIZE the score (from White's perspective)
        best_value = float('-inf')
        for move in board.legal_moves:
            board.push(move)
             
            """ADDED CHECK FOR REPETITION HERE!!!!!!!!"""
            # Penalize moves that lead to threefold repetition
            # Since all scores are from White's perspective, repetition is bad for the player
            if board.is_repetition(2):  #checks if the position has occurred 2+ times (would be 3rd occurrence)
                board_value = -900  # Large penalty from White's perspective
            else:
                board_value = minimax(board, depth - 1, alpha, beta, False)
            
            board.pop()
            
            if board_value > best_value:
                best_value = board_value
                best_move = move
            alpha = max(alpha, best_value)
    else:
        # Black wants to MINIMIZE the score (from White's perspective)
        best_value = float('inf')
        for move in board.legal_moves:
            board.push(move)
            
            # For Black, repetition should also be avoided
            # From White's perspective, if Black repeats, it's like a draw offer (close to 0)
            # But Black wants to avoid it, so we still penalize it
            if board.is_repetition(2):  
                board_value = 900  # Large bonus from White's perspective (bad for Black)
            else:
                board_value = minimax(board, depth - 1, alpha, beta, True)
            
            board.pop()
            
            if board_value < best_value:
                best_value = board_value
                best_move = move
            beta = min(beta, best_value)
    
    return best_move

def play(interface: Interface, color = "w"):
    search_depth = 3  # Can be any positive number
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