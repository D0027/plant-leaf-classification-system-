from flask import Flask, render_template, jsonify, request, session, redirect
import math
import heapq
import random
import itertools
from collections import deque
from textblob import TextBlob
import re

# ── ADDED IMPORTS ────────────────────────────────────────────────────────────
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, Score, Achievement
import os

app = Flask(__name__)

# ── AUTH / DB CONFIG ─────────────────────────────────────────────────────────
app.secret_key = os.environ.get('SECRET_KEY', 'neural-arena-dev-key-change-in-prod')

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    'DATABASE_URL', 'sqlite:///neural_arena.db'
)
# Render gives postgres://, SQLAlchemy needs postgresql://
if app.config['SQLALCHEMY_DATABASE_URI'].startswith('postgres://'):
    app.config['SQLALCHEMY_DATABASE_URI'] = \
        app.config['SQLALCHEMY_DATABASE_URI'].replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# --- ROUTES ---
@app.route('/')
def dashboard(): return render_template('dashboard.html')

@app.route('/tictactoe')
def tictactoe(): return render_template('tictactoe.html')

@app.route('/pacman')
def pacman(): return render_template('pacman.html')

@app.route('/rps')
def rps(): return render_template('rps.html')

@app.route('/connect4')
def connect4_route(): return render_template('connect4.html')

@app.route('/pong')
def pong(): return render_template('pong.html')

@app.route('/snake')
def snake(): return render_template('snake.html')

@app.route('/2048')
def game_2048(): return render_template('2048.html')

@app.route('/maze')
def maze(): return render_template('maze.html')

@app.route('/mines')
def mines(): return render_template('mines.html')

@app.route('/breakout')
def breakout(): return render_template('breakout.html')

@app.route('/tetris')
def tetris(): return render_template('tetris.html')

@app.route('/flappy')
def flappy(): return render_template('flappy.html')

@app.route('/invaders')
def invaders(): return render_template('invaders.html')

@app.route('/runner')
def runner(): return render_template('runner.html')

@app.route('/guesswho')
def guesswho(): return render_template('guesswho.html')

@app.route('/wordle')
def wordle(): return render_template('wordle.html')

@app.route('/sudoku')
def sudoku(): return render_template('sudoku.html')

@app.route('/memory')
def memory(): return render_template('memory.html')

@app.route('/hanoi')
def hanoi(): return render_template('hanoi.html')

@app.route('/network')
def network(): return render_template('network.html')

@app.route('/path')
def path(): return render_template('path.html')

@app.route('/slide')
def slide(): return render_template('slide.html')

@app.route('/cipher')
def cipher(): return render_template('cipher.html')

@app.route('/switch')
def switch(): return render_template('switch.html')

@app.route('/knight')
def knight(): return render_template('knight.html')

@app.route('/cluster')
def cluster(): return render_template('cluster.html')

@app.route('/life')
def life(): return render_template('life.html')

@app.route('/evo')
def evo(): return render_template('evo.html')

@app.route('/regress')
def regress(): return render_template('regress.html')

@app.route('/sort')
def sort_game(): return render_template('sort.html')

@app.route('/pack')
def pack(): return render_template('pack.html')

@app.route('/hull')
def hull(): return render_template('hull.html')

@app.route('/span')
def span(): return render_template('span.html')

@app.route('/flow')
def flow(): return render_template('flow.html')

@app.route('/color')
def color_game(): return render_template('color.html')

@app.route('/queen')
def queen(): return render_template('queen.html')

@app.route('/match')
def match(): return render_template('match.html')


# ==========================================
# GAME 1: TIC-TAC-TOE (MINIMAX)
# ==========================================
def check_winner(board):
    winning_combos = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
    for combo in winning_combos:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] and board[combo[0]] != "":
            return board[combo[0]]
    if "" not in board: return "Draw"
    return None

def minimax(board, depth, is_maximizing):
    winner = check_winner(board)
    if winner == 'O': return 10 - depth
    if winner == 'X': return -10 + depth
    if winner == 'Draw': return 0
    if is_maximizing:
        best_score = -math.inf
        for i in range(9):
            if board[i] == "":
                board[i] = 'O'; score = minimax(board, depth + 1, False); board[i] = ""
                best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for i in range(9):
            if board[i] == "":
                board[i] = 'X'; score = minimax(board, depth + 1, True); board[i] = ""
                best_score = min(score, best_score)
        return best_score

@app.route('/move', methods=['POST'])
def move():
    data = request.json; board = data['board']
    if check_winner(board): return jsonify({'index': None, 'game_over': True})
    best_score = -math.inf; best_move = None
    for i in range(9):
        if board[i] == "":
            board[i] = 'O'; score = minimax(board, 0, False); board[i] = ""
            if score > best_score: best_score = score; best_move = i
    return jsonify({'index': best_move})


# ==========================================
# GAME 2: PAC-MAN (A* PATHFINDING)
# ==========================================
class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent; self.position = position
        self.g = 0; self.h = 0; self.f = 0
    def __eq__(self, other): return self.position == other.position
    def __lt__(self, other): return self.f < other.f

def astar(maze, start, end):
    start_node = Node(None, tuple(start)); end_node = Node(None, tuple(end))
    open_list = []; closed_list = set()
    heapq.heappush(open_list, start_node)
    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node.position)
        if current_node == end_node:
            path = []; current = current_node
            while current is not None:
                path.append(current.position); current = current.parent
            return path[::-1]
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            node_position = (
                current_node.position[0] + new_position[0],
                current_node.position[1] + new_position[1]
            )
            if (node_position[0] > len(maze) - 1 or node_position[0] < 0 or
                    node_position[1] > len(maze[0]) - 1 or node_position[1] < 0):
                continue
            if maze[node_position[0]][node_position[1]] == 1: continue
            if node_position in closed_list: continue
            new_node = Node(current_node, node_position)
            new_node.g = current_node.g + 1
            new_node.h = (abs(new_node.position[0] - end_node.position[0]) +
                          abs(new_node.position[1] - end_node.position[1]))
            new_node.f = new_node.g + new_node.h
            heapq.heappush(open_list, new_node)
    return None

@app.route('/move_pacman', methods=['POST'])
def move_pacman():
    data = request.json
    maze = data['grid']; ghosts = data['ghosts']
    player = data['playerPos']; goal = data['goalPos']
    new_ghost_positions = []; all_paths = []

    # Hunter ghost
    path_hunter = astar(maze, ghosts[0], player)
    if path_hunter and len(path_hunter) > 1:
        new_ghost_positions.append(path_hunter[1]); all_paths.append(path_hunter)
    else:
        new_ghost_positions.append(ghosts[0])

    # Tactician ghost
    dist_player_goal = abs(player[0] - goal[0]) + abs(player[1] - goal[1])
    target = player if dist_player_goal < 8 else goal
    path_tactician = astar(maze, ghosts[1], target)
    if path_tactician and len(path_tactician) > 1:
        if path_tactician[1][0] == goal[0] and path_tactician[1][1] == goal[1]:
            new_ghost_positions.append(ghosts[1])
        else:
            new_ghost_positions.append(path_tactician[1])
        all_paths.append(path_tactician)
    else:
        new_ghost_positions.append(ghosts[1])

    return jsonify({
        'ghosts': new_ghost_positions,
        'paths': all_paths,
        'hunter_stuck': path_hunter is None,
        'tactician_stuck': path_tactician is None
    })


# ==========================================
# GAME 3: RPS (MULTI-DIMENSIONAL MARKOV)
# ==========================================

@app.route('/reset_rps', methods=['POST'])
def reset_rps():
    session['rps_history'] = ""
    session['rps_m3'] = {}
    session['rps_m4'] = {}
    session['rps_m5'] = {}
    return jsonify({'status': 'reset'})

@app.route('/move_rps', methods=['POST'])
def move_rps():
    user_history = session.get('rps_history', "")
    matrix_3 = session.get('rps_m3', {})
    matrix_4 = session.get('rps_m4', {})
    matrix_5 = session.get('rps_m5', {})

    data = request.json; user_move = data['move']
    predicted_user_move = None; confidence = 0
    potential_moves = ['R', 'P', 'S']

    if not predicted_user_move and len(user_history) >= 4:
        last_4 = user_history[-4:]
        counts = {m: matrix_5.get(last_4 + m, 0) for m in potential_moves}
        if sum(counts.values()) > 0:
            predicted_user_move = max(counts, key=counts.get); confidence = 90

    if not predicted_user_move and len(user_history) >= 3:
        last_3 = user_history[-3:]
        counts = {m: matrix_4.get(last_3 + m, 0) for m in potential_moves}
        if sum(counts.values()) > 0:
            predicted_user_move = max(counts, key=counts.get); confidence = 75

    if not predicted_user_move and len(user_history) >= 2:
        last_2 = user_history[-2:]
        counts = {m: matrix_3.get(last_2 + m, 0) for m in potential_moves}
        if sum(counts.values()) > 0:
            predicted_user_move = max(counts, key=counts.get); confidence = 50

    if not predicted_user_move:
        predicted_user_move = random.choice(potential_moves); confidence = 10

    counter = {'R': 'P', 'P': 'S', 'S': 'R'}
    ai_move = counter[predicted_user_move]

    if len(user_history) >= 4:
        key = user_history[-4:] + user_move
        matrix_5[key] = matrix_5.get(key, 0) + 1
    if len(user_history) >= 3:
        key = user_history[-3:] + user_move
        matrix_4[key] = matrix_4.get(key, 0) + 1
    if len(user_history) >= 2:
        key = user_history[-2:] + user_move
        matrix_3[key] = matrix_3.get(key, 0) + 1
    user_history += user_move

    session['rps_history'] = user_history
    session['rps_m3'] = matrix_3
    session['rps_m4'] = matrix_4
    session['rps_m5'] = matrix_5

    if user_move == ai_move:
        result = "DRAW"
    elif (user_move == 'R' and ai_move == 'S') or \
         (user_move == 'P' and ai_move == 'R') or \
         (user_move == 'S' and ai_move == 'P'):
        result = "YOU WIN"
    else:
        result = "AI WINS"

    return jsonify({'ai_move': ai_move, 'result': result,
                    'prediction': predicted_user_move, 'confidence': confidence})


# ==========================================
# GAME 4: CONNECT 4 (ALPHA-BETA PRUNING)
# ==========================================
ROW_COUNT = 6
COLUMN_COUNT = 7

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
    return board[0][col] == 0

def get_next_open_row(board, col):
    for r in range(ROW_COUNT - 1, -1, -1):
        if board[r][col] == 0: return r

def winning_move(board, piece):
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if (board[r][c] == piece and board[r][c+1] == piece and
                    board[r][c+2] == piece and board[r][c+3] == piece):
                return True
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if (board[r][c] == piece and board[r+1][c] == piece and
                    board[r+2][c] == piece and board[r+3][c] == piece):
                return True
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if (board[r][c] == piece and board[r+1][c+1] == piece and
                    board[r+2][c+2] == piece and board[r+3][c+3] == piece):
                return True
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if (board[r][c] == piece and board[r-1][c+1] == piece and
                    board[r-2][c+2] == piece and board[r-3][c+3] == piece):
                return True
    return False

def evaluate_window(window, piece):
    score = 0; opp_piece = 1 if piece == 2 else 2
    if window.count(piece) == 4: score += 100
    elif window.count(piece) == 3 and window.count(0) == 1: score += 5
    elif window.count(piece) == 2 and window.count(0) == 2: score += 2
    if window.count(opp_piece) == 3 and window.count(0) == 1: score -= 4
    return score

def score_position(board, piece):
    score = 0
    center_array = [row[COLUMN_COUNT // 2] for row in board]
    score += center_array.count(piece) * 3
    for r in range(ROW_COUNT):
        row_array = board[r]
        for c in range(COLUMN_COUNT - 3):
            score += evaluate_window(row_array[c:c+4], piece)
    for c in range(COLUMN_COUNT):
        col_array = [board[r][c] for r in range(ROW_COUNT)]
        for r in range(ROW_COUNT - 3):
            score += evaluate_window(col_array[r:r+4], piece)
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            score += evaluate_window([board[r+i][c+i] for i in range(4)], piece)
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            score += evaluate_window([board[r+3-i][c+i] for i in range(4)], piece)
    return score

def get_valid_locations(board):
    return [col for col in range(COLUMN_COUNT) if is_valid_location(board, col)]

def is_terminal_node(board):
    return winning_move(board, 1) or winning_move(board, 2) or len(get_valid_locations(board)) == 0

def minimax_c4(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, 2): return (None, 100000000000000)
            elif winning_move(board, 1): return (None, -10000000000000)
            else: return (None, 0)
        else:
            return (None, score_position(board, 2))
    if maximizingPlayer:
        value = -math.inf; column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = [r[:] for r in board]
            drop_piece(b_copy, row, col, 2)
            new_score = minimax_c4(b_copy, depth - 1, alpha, beta, False)[1]
            if new_score > value: value = new_score; column = col
            alpha = max(alpha, value)
            if alpha >= beta: break
        return column, value
    else:
        value = math.inf; column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = [r[:] for r in board]
            drop_piece(b_copy, row, col, 1)
            new_score = minimax_c4(b_copy, depth - 1, alpha, beta, True)[1]
            if new_score < value: value = new_score; column = col
            beta = min(beta, value)
            if alpha >= beta: break
        return column, value

@app.route('/move_connect4', methods=['POST'])
def move_connect4():
    data = request.json; board = data['board']
    col, _ = minimax_c4(board, 4, -math.inf, math.inf, True)
    if col is not None:
        row = get_next_open_row(board, col)
        return jsonify({'col': col, 'row': row})
    return jsonify({'col': None})


# ==========================================
# GAME 6: NEON SNAKE (BFS AUTOPILOT)
# ==========================================
def bfs_snake(grid_size, snake_body, food):
    start = tuple(snake_body[0]); target = tuple(food)
    obstacles = set(tuple(x) for x in snake_body)
    queue = [(start, [])]; visited = {start}
    while queue:
        (current, path) = queue.pop(0)
        if current == target:
            return path[0] if path else None
        x, y = current
        for nx, ny in [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]:
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                if (nx, ny) not in obstacles or (nx, ny) == target:
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append(((nx, ny), path + [(nx, ny)]))
    return None

@app.route('/move_snake_ai', methods=['POST'])
def move_snake_ai():
    data = request.json
    grid_size = data['gridSize']; snake = data['snake']; food = data['food']
    next_step = bfs_snake(grid_size, snake, food)
    direction = None

    if not next_step:
        head_x, head_y = snake[0]
        obstacles = set(tuple(x) for x in snake[:-1])
        for nx, ny in [(head_x+1, head_y), (head_x-1, head_y),
                       (head_x, head_y+1), (head_x, head_y-1)]:
            if 0 <= nx < grid_size and 0 <= ny < grid_size and (nx, ny) not in obstacles:
                next_step = (nx, ny)
                break

    if next_step:
        head_x, head_y = snake[0]; next_x, next_y = next_step
        if next_x > head_x: direction = 'RIGHT'
        elif next_x < head_x: direction = 'LEFT'
        elif next_y > head_y: direction = 'DOWN'
        elif next_y < head_y: direction = 'UP'

    return jsonify({'direction': direction})


# ==========================================
# GAME 7: NEON 2048 (EXPECTIMAX AI)
# ==========================================
def get_empty_cells(grid):
    return [(r, c) for r in range(4) for c in range(4) if grid[r][c] == 0]

def merge_line(line):
    new_line = [i for i in line if i != 0]
    for i in range(len(new_line) - 1):
        if new_line[i] == new_line[i+1]:
            new_line[i] *= 2; new_line[i+1] = 0
    new_line = [i for i in new_line if i != 0]
    return new_line + [0] * (4 - len(new_line))

def simulate_move(grid, direction):
    new_grid = [row[:] for row in grid]
    if direction == 0:
        new_grid = [list(x) for x in zip(*new_grid)]
        new_grid = [merge_line(row) for row in new_grid]
        new_grid = [list(x) for x in zip(*new_grid)]
    elif direction == 1:
        new_grid = [list(x) for x in zip(*new_grid)]
        new_grid = [merge_line(row[::-1])[::-1] for row in new_grid]
        new_grid = [list(x) for x in zip(*new_grid)]
    elif direction == 2:
        new_grid = [merge_line(row) for row in new_grid]
    elif direction == 3:
        new_grid = [merge_line(row[::-1])[::-1] for row in new_grid]
    return new_grid

def grid_score(grid):
    empty = len(get_empty_cells(grid))
    weights = [[65536,32768,16384,8192],[512,1024,2048,4096],[256,128,64,32],[2,4,8,16]]
    score = sum(grid[r][c] * weights[r][c] for r in range(4) for c in range(4))
    return score + (empty * 1000)

def expectimax(grid, depth, is_player):
    if depth == 0: return grid_score(grid)
    if is_player:
        best_score = -math.inf
        for move in range(4):
            sim_grid = simulate_move(grid, move)
            if sim_grid != grid:
                score = expectimax(sim_grid, depth - 1, False)
                best_score = max(best_score, score)
        return best_score if best_score != -math.inf else 0
    else:
        empty = get_empty_cells(grid)
        if not empty: return grid_score(grid)
        check_spots = empty[:2] if len(empty) > 2 else empty
        avg_score = 0
        for r, c in check_spots:
            temp2 = [row[:] for row in grid]
            temp4 = [row[:] for row in grid]
            temp2[r][c] = 2
            temp4[r][c] = 4
            avg_score += 0.9 * expectimax(temp2, depth - 1, True)
            avg_score += 0.1 * expectimax(temp4, depth - 1, True)
        return avg_score / len(check_spots)

@app.route('/move_2048', methods=['POST'])
def move_2048():
    grid = request.json['grid']
    best_move = -1; best_score = -math.inf
    for move in range(4):
        sim_grid = simulate_move(grid, move)
        if sim_grid != grid:
            score = expectimax(sim_grid, 3, False)
            if score > best_score:
                best_score = score; best_move = move
    moves = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    return jsonify({'move': moves[best_move] if best_move != -1 else None})


# ==========================================
# GAME 8: NEON MAZE (DFS SOLVER)
# ==========================================
def solve_dfs(grid, start, end):
    stack = [(start, [start])]; visited = set()
    while stack:
        (vertex, path) = stack.pop()
        if vertex not in visited:
            if vertex == end: return path
            visited.add(vertex)
            r, c = vertex
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] == 0:
                    stack.append(((nr, nc), path + [(nr, nc)]))
    return []

@app.route('/solve_maze', methods=['POST'])
def solve_maze():
    data = request.json; grid = data['grid']
    start = tuple(data['start']); end = tuple(data['end'])
    path = solve_dfs(grid, start, end)
    return jsonify({'path': path})


# ==========================================
# GAME 9: NEON MINES (CONSTRAINT SOLVER)
# ==========================================
@app.route('/solve_mines', methods=['POST'])
def solve_mines():
    grid = request.json['grid']
    rows = len(grid); cols = len(grid[0])
    moves = []
    decided = set()

    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            if val >= 1:
                hidden = []; flagged = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            state = grid[nr][nc]
                            if state == -1: hidden.append((nr, nc))
                            elif state == -2: flagged.append((nr, nc))

                if len(flagged) == val and hidden:
                    for hr, hc in hidden:
                        if (hr, hc) not in decided:
                            decided.add((hr, hc))
                            moves.append({'r': hr, 'c': hc, 'action': 'REVEAL'})

                if len(hidden) + len(flagged) == val and hidden:
                    for hr, hc in hidden:
                        if (hr, hc) not in decided:
                            decided.add((hr, hc))
                            moves.append({'r': hr, 'c': hc, 'action': 'FLAG'})

    return jsonify({'moves': moves})


# ==========================================
# GAME 10: NEON BREAKOUT (REFLEX AGENT)
# ==========================================
@app.route('/move_breakout', methods=['POST'])
def move_breakout():
    data = request.json
    paddle_x = data.get('paddle_x'); paddle_w = data.get('paddle_width')
    ball_x = data.get('ball_x'); ball_dy = data.get('ball_dy')
    width = data.get('canvas_width')
    paddle_center = paddle_x + paddle_w / 2
    target_x = ball_x if ball_dy > 0 else width / 2
    if target_x < paddle_center - 10: return jsonify({'move': 'LEFT'})
    elif target_x > paddle_center + 10: return jsonify({'move': 'RIGHT'})
    else: return jsonify({'move': 'STOP'})


# ==========================================
# GAME 11: NEON TETRIS (HEURISTIC SEARCH)
# ==========================================
SHAPES = {
    'I': [[1, 1, 1, 1]],
    'J': [[1, 0, 0], [1, 1, 1]],
    'L': [[0, 0, 1], [1, 1, 1]],
    'O': [[1, 1], [1, 1]],
    'S': [[0, 1, 1], [1, 1, 0]],
    'T': [[0, 1, 0], [1, 1, 1]],
    'Z': [[1, 1, 0], [0, 1, 1]]
}

def rotate_shape(shape, times):
    s = [row[:] for row in shape]
    for _ in range(times):
        s = [list(row) for row in zip(*s[::-1])]
    return s

def check_collision(board, shape, offset):
    off_x, off_y = offset
    for y, row in enumerate(shape):
        for x, val in enumerate(row):
            if val:
                if (y + off_y >= len(board) or x + off_x < 0 or
                        x + off_x >= len(board[0]) or
                        (y + off_y >= 0 and board[y + off_y][x + off_x])):
                    return True
    return False

def get_landing_height(board, shape, x):
    y = 0
    while not check_collision(board, shape, (x, y)):
        y += 1
    return y - 1

def score_board(board):
    heights = []; holes = 0; lines_cleared = 0
    for c in range(len(board[0])):
        h = 0
        for r in range(len(board)):
            if board[r][c]:
                h = len(board) - r; break
        heights.append(h)
        for r in range(len(board) - h, len(board)):
            if board[r][c] == 0: holes += 1
    agg_height = sum(heights)
    bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(len(heights) - 1))
    for r in range(len(board)):
        if all(board[r]): lines_cleared += 1
    return (-0.51 * agg_height) + (0.76 * lines_cleared) - (0.36 * holes) - (0.18 * bumpiness)

@app.route('/solve_tetris', methods=['POST'])
def solve_tetris():
    data = request.json; board = data['grid']; piece_type = data['piece']
    best_score = -math.inf; best_move = {'rotation': 0, 'x': 0}
    for r in range(4):
        shape = rotate_shape(SHAPES[piece_type], r)
        width = len(shape[0])
        for x in range(0, len(board[0]) - width + 1):
            y = get_landing_height(board, shape, x)
            if y < 0: continue
            temp_board = [row[:] for row in board]; valid = True
            for sy, row in enumerate(shape):
                for sx, val in enumerate(row):
                    if val:
                        if 0 <= y + sy < len(temp_board) and 0 <= x + sx < len(temp_board[0]):
                            temp_board[y + sy][x + sx] = 1
                        else:
                            valid = False
            if valid:
                score = score_board(temp_board)
                if score > best_score:
                    best_score = score; best_move = {'rotation': r, 'x': x}
    return jsonify(best_move)


# ==========================================
# GAME 12: NEON FLAPPY (NEURAL DECISION)
# ==========================================
@app.route('/move_flappy', methods=['POST'])
def move_flappy():
    data = request.json
    bird_y = data['birdY']; gap_y = data['gapY']; gap_height = data['gapHeight']
    target_y = gap_y + (gap_height / 2) + 15
    return jsonify({'jump': bird_y > target_y})


# ==========================================
# GAME 15: NEON PROPHET (DECISION TREE)
# ==========================================
decision_tree = {
    'root':         {'q': "Is your character ORGANIC (living tissue)?", 'yes': 'organic', 'no': 'synthetic'},
    'organic':      {'q': "Does this character use WEAPONS?", 'yes': 'soldier_path', 'no': 'civilian_path'},
    'synthetic':    {'q': "Does it look HUMAN (Android)?", 'yes': 'android_path', 'no': 'machine_path'},
    'soldier_path': {'q': "Do they prefer STEALTH over brute force?", 'yes': 'THE NINJA', 'no': 'THE SOLDIER'},
    'civilian_path':{'q': "Are they a criminal (Hacker)?", 'yes': 'THE HACKER', 'no': 'THE CIVILIAN'},
    'android_path': {'q': "Is it programmed to KILL?", 'yes': 'THE TERMINATOR', 'no': 'THE BUTLER BOT'},
    'machine_path': {'q': "Can it FLY?", 'yes': 'THE DRONE', 'no': 'THE AI CORE'}
}

@app.route('/query_tree', methods=['POST'])
def query_tree():
    data = request.json
    current_node = data.get('node', 'root'); answer = data.get('answer')
    if answer is None:
        return jsonify({'node': 'root', 'text': decision_tree['root']['q'], 'is_guess': False})
    if current_node in decision_tree:
        next_step = decision_tree[current_node].get(answer)
        if next_step not in decision_tree:
            return jsonify({'node': next_step, 'text': f"Is it... {next_step}?", 'is_guess': True})
        else:
            return jsonify({'node': next_step, 'text': decision_tree[next_step]['q'], 'is_guess': False})
    return jsonify({'error': 'Neural Link Lost'})


# ==========================================
# GAME 16: NEON WORDLE (ENTROPY SOLVER)
# ==========================================
WORD_LIST = [
    "REACT", "ADIEU", "LATER", "SIREN", "TEARS", "ALONE", "ARISE", "SMART", "STARE", "STORY",
    "RAISE", "ROAST", "CRANE", "SLATE", "TRACE", "AUDIO", "RADIO", "STONE", "EARTH", "MEDIA",
    "SPEED", "STEAL", "TRADE", "MODEL", "BLOCK", "BROWN", "BUILD", "CAUSE", "CHECK", "CHEST",
    "CLAIM", "CLASS", "CLEAN", "CLEAR", "CLIMB", "CLOCK", "CLOSE", "COACH", "COAST", "COUNT",
    "COURT", "COVER", "CROSS", "CROWD", "CROWN", "CYCLE", "DANCE", "DEATH", "DEPTH", "DOUBT",
    "DRAFT", "DRAMA", "DREAM", "DRESS", "DRINK", "DRIVE", "EARLY", "ENTRY", "EQUAL", "ERROR",
    "EVENT", "EXACT", "EXIST", "FAITH", "FAULT", "FIBER", "FIELD", "FINAL", "FLASH", "FLOOR",
    "FOCUS", "FORCE", "FRAME", "FRANK", "FRONT", "FRUIT", "GLASS", "GRANT", "GRASS", "GREEN",
    "GROUP", "GUIDE", "HEART", "HEAVY", "HORSE", "HOTEL", "HOUSE", "IMAGE", "INDEX", "INPUT",
    "ISSUE", "JUDGE", "KNIFE", "LAYER", "LEVEL", "LIGHT", "LIMIT", "LOCAL", "LOGIC", "LUNCH",
    "MAJOR", "MARCH", "MATCH", "METAL", "MODEL", "MONEY", "MONTH", "MOTOR", "MOUTH", "MUSIC",
    "NIGHT", "NOISE", "NORTH", "NOVEL", "NURSE", "OFFER", "ORDER", "OTHER", "OWNER", "PANEL",
    "PAPER", "PARTY", "PEACE", "PHASE", "PHONE", "PIECE", "PILOT", "PITCH", "PLACE", "PLANE",
    "PLANT", "PLATE", "POINT", "POUND", "POWER", "PRESS", "PRICE", "PRIDE", "PRIZE", "PROOF",
    "QUEEN", "RADIO", "RANGE", "RATIO", "REPLY", "RIGHT", "RIVER", "ROUND", "ROUTE", "RUGBY",
    "SCALE", "SCENE", "SCOPE", "SCORE", "SENSE", "SHAPE", "SHARE", "SHEEP", "SHEET", "SHIFT",
    "SHIRT", "SHOCK", "SHOOT", "SHORT", "SIGHT", "SKILL", "SLEEP", "SMALL", "SMILE", "SMOKE",
    "SOLID", "SOLVE", "SOUND", "SOUTH", "SPACE", "SPARE", "SPEAK", "SPEED", "SPITE", "SPORT",
    "SQUAD", "STACK", "STAFF", "STAGE", "STAND", "START", "STATE", "STEAM", "STEEL", "STICK",
    "STILL", "STOCK", "STONE", "STORE", "STUDY", "STUFF", "STYLE", "SUGAR", "TABLE", "TASTE",
    "TEACH", "THANK", "THEME", "THING", "THINK", "THROW", "TITLE", "TOTAL", "TOUCH", "TOWER",
    "TRACK", "TRADE", "TRAIN", "TREAT", "TRUCK", "TRUST", "TRUTH", "UNCLE", "UNION", "UNITY",
    "VALUE", "VIDEO", "VISIT", "VOICE", "WASTE", "WATCH", "WATER", "WHILE", "WHITE", "WHOLE",
    "WOMAN", "WORLD", "WORRY", "WRITE", "WRONG", "YOUTH", "ZEBRA", "ROBOT", "ALIEN", "LASER"
]

def check_word_validity(word, constraints):
    for c in constraints:
        if c['status'] == 'correct':
            if word[c['index']] != c['letter']: return False
    for c in constraints:
        if c['status'] == 'absent':
            required = any(x['letter'] == c['letter'] and x['status'] in ['correct', 'present']
                           for x in constraints)
            if not required and c['letter'] in word: return False
    for c in constraints:
        if c['status'] == 'present':
            if c['letter'] not in word: return False
            if word[c['index']] == c['letter']: return False
    return True

@app.route('/solve_wordle', methods=['POST'])
def solve_wordle():
    data = request.json; constraints = data['constraints']
    try:
        candidates = [w for w in WORD_LIST if check_word_validity(w, constraints)]
    except (KeyError, IndexError):
        return jsonify({'guess': 'RESET', 'candidates_left': 0})
    if not candidates:
        return jsonify({'guess': "RESET"})
    if len(constraints) == 0:
        best_word = "CRANE"
    else:
        best_word = max(candidates, key=lambda w: len(set(w)))
    return jsonify({'guess': best_word, 'candidates_left': len(candidates)})


# ==========================================
# GAME 17: NEON SUDOKU (BACKTRACKING)
# ==========================================
def is_valid_sudoku(board, r, c, num):
    if num in board[r]: return False
    if num in [board[x][c] for x in range(9)]: return False
    start_r, start_c = 3 * (r // 3), 3 * (c // 3)
    for i in range(3):
        for j in range(3):
            if board[start_r + i][start_c + j] == num: return False
    return True

def solve_sudoku_algo(board):
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                for num in range(1, 10):
                    if is_valid_sudoku(board, r, c, num):
                        board[r][c] = num
                        if solve_sudoku_algo(board): return True
                        board[r][c] = 0
                return False
    return True

@app.route('/solve_sudoku', methods=['POST'])
def solve_sudoku():
    data = request.json; board = data['grid']
    if solve_sudoku_algo(board):
        return jsonify({'solved': True, 'board': board})
    return jsonify({'solved': False})


# ==========================================
# GAME 18: NEON CORTEX (MEMORY MATCH)
# ==========================================
@app.route('/solve_memory', methods=['POST'])
def solve_memory():
    data = request.json
    memory = data['memory']; hidden_indices = data['hidden']; true_board = data['board']
    counts = {}
    for idx, val in memory.items():
        idx = int(idx)
        if idx in hidden_indices:
            if val in counts: counts[val].append(idx)
            else: counts[val] = [idx]
    for val, indices in counts.items():
        if len(indices) == 2:
            return jsonify({'move': indices, 'reason': 'recall'})
    guess1 = random.choice(hidden_indices); val1 = true_board[guess1]
    match_in_memory = -1
    for idx, val in memory.items():
        if val == val1 and int(idx) != guess1 and int(idx) in hidden_indices:
            match_in_memory = int(idx); break
    if match_in_memory != -1:
        return jsonify({'move': [guess1, match_in_memory], 'reason': 'lucky_match'})
    remaining = [x for x in hidden_indices if x != guess1]
    guess2 = random.choice(remaining)
    return jsonify({'move': [guess1, guess2], 'reason': 'guess'})


# ==========================================
# GAME 19: NEON HANOI (RECURSION)
# ==========================================
@app.route('/solve_hanoi', methods=['POST'])
def solve_hanoi():
    n = request.json.get('disks', 5); moves = []
    def hanoi_algo(n, source, target, auxiliary):
        if n > 0:
            hanoi_algo(n - 1, source, auxiliary, target)
            moves.append({'from': source, 'to': target})
            hanoi_algo(n - 1, auxiliary, target, source)
    hanoi_algo(n, 0, 2, 1)
    return jsonify(moves)


# ==========================================
# GAME 20: NEON NETWORK (TSP OPTIMIZER)
# ==========================================
@app.route('/solve_tsp', methods=['POST'])
def solve_tsp():
    points = request.json['points']; path = list(range(len(points)))
    def get_dist(i, j):
        p1 = points[path[i]]; p2 = points[path[j]]
        return math.hypot(p1['x'] - p2['x'], p1['y'] - p2['y'])
    improved = True
    while improved:
        improved = False
        for i in range(len(path) - 1):
            for j in range(i + 2, len(path)):
                if j == len(path) - 1 and i == 0: continue
                a, b = i, (i + 1) % len(path)
                c, d = j, (j + 1) % len(path)
                if get_dist(a, b) + get_dist(c, d) > get_dist(a, c) + get_dist(b, d):
                    path[i+1:j+1] = path[i+1:j+1][::-1]
                    improved = True
    return jsonify({'path': path})


# ==========================================
# GAME 21: NEON PATH (BFS PATHFINDER)
# ==========================================
@app.route('/solve_path', methods=['POST'])
def solve_path():
    grid = request.json['grid']
    rows = len(grid); cols = len(grid[0])
    start = (0, 0); end = (rows - 1, cols - 1)

    queue = deque([start])
    visited = {start: None}

    while queue:
        curr = queue.popleft()
        if curr == end: break
        row, col = curr
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr][nc] == 0 and (nr, nc) not in visited:
                    visited[(nr, nc)] = curr
                    queue.append((nr, nc))

    path_coords = []
    if end in visited:
        curr = end
        while curr is not None:
            path_coords.append({'row': curr[0], 'col': curr[1]})
            curr = visited[curr]
        path_coords.reverse()

    return jsonify({'path': path_coords})


# ==========================================
# GAME 22: NEON SLIDE (8-PUZZLE A* SOLVER)
# ==========================================
@app.route('/solve_slide', methods=['POST'])
def solve_slide():
    start_grid = tuple(request.json['grid'])
    goal_grid = (1, 2, 3, 4, 5, 6, 7, 8, 0)

    def manhattan(state):
        dist = 0
        for i, val in enumerate(state):
            if val == 0: continue
            target_idx = val - 1
            cur_r, cur_c = divmod(i, 3)
            tar_r, tar_c = divmod(target_idx, 3)
            dist += abs(cur_r - tar_r) + abs(cur_c - tar_c)
        return dist

    pq = [(manhattan(start_grid), 0, start_grid, [])]
    visited = set()
    while pq:
        f, g, current, path = heapq.heappop(pq)
        if current == goal_grid: return jsonify({'moves': path})
        if current in visited: continue
        visited.add(current)
        zero_idx = current.index(0)
        z_row, z_col = divmod(zero_idx, 3)
        for dr, dc, move_name in [(-1, 0, 'U'), (1, 0, 'D'), (0, -1, 'L'), (0, 1, 'R')]:
            nr, nc = z_row + dr, z_col + dc
            if 0 <= nr < 3 and 0 <= nc < 3:
                new_idx = nr * 3 + nc
                new_state = list(current)
                new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
                new_state = tuple(new_state)
                if new_state not in visited:
                    new_g = g + 1
                    heapq.heappush(pq, (new_g + manhattan(new_state), new_g, new_state, path + [new_idx]))
    return jsonify({'moves': []})


# ==========================================
# GAME 23: NEON CIPHER (MASTERMIND SOLVER)
# ==========================================
@app.route('/solve_cipher', methods=['POST'])
def solve_cipher():
    history = request.json['history']
    all_codes = list(itertools.product(range(6), repeat=4))

    def get_feedback(guess, code):
        black = 0; white = 0
        code_counts = [0] * 6; guess_counts = [0] * 6
        for i in range(4):
            if guess[i] == code[i]: black += 1
            else: code_counts[code[i]] += 1; guess_counts[guess[i]] += 1
        for i in range(6):
            white += min(code_counts[i], guess_counts[i])
        return {'black': black, 'white': white}

    candidates = []
    for code in all_codes:
        is_valid = True
        for turn in history:
            past_guess = tuple(turn['guess'])
            if get_feedback(past_guess, code) != turn['feedback']:
                is_valid = False; break
        if is_valid: candidates.append(code)

    if candidates:
        return jsonify({'suggestion': candidates[0], 'candidates_left': len(candidates)})
    return jsonify({'suggestion': None, 'candidates_left': 0})


# ==========================================
# GAME 24: NEON SWITCH (LIGHTS OUT SOLVER)
# ==========================================
@app.route('/solve_switch', methods=['POST'])
def solve_switch():
    rows = 5; cols = 5
    original_grid = [row[:] for row in request.json['grid']]
    grid = [row[:] for row in original_grid]
    moves = []

    def toggle(r, c, g):
        if 0 <= r < rows and 0 <= c < cols:
            g[r][c] = 1 - g[r][c]

    def apply_move(r, c, g, m):
        toggle(r, c, g); toggle(r-1, c, g); toggle(r+1, c, g)
        toggle(r, c-1, g); toggle(r, c+1, g)
        m.append({'r': r, 'c': c})

    for r in range(rows - 1):
        for c in range(cols):
            if grid[r][c] == 1:
                apply_move(r + 1, c, grid, moves)

    if any(grid[rows - 1][c] == 1 for c in range(cols)):
        for mask in range(32):
            test_grid = [row[:] for row in original_grid]
            test_moves = []

            for c in range(cols):
                if mask & (1 << c):
                    apply_move(0, c, test_grid, test_moves)

            for r in range(rows - 1):
                for c in range(cols):
                    if test_grid[r][c] == 1:
                        apply_move(r + 1, c, test_grid, test_moves)

            if all(test_grid[r][c] == 0 for r in range(rows) for c in range(cols)):
                return jsonify({'moves': test_moves})

    return jsonify({'moves': moves})


# ==========================================
# GAME 25: NEON KNIGHT (WARNSDORFF'S RULE)
# ==========================================
@app.route('/solve_knight', methods=['POST'])
def solve_knight():
    start_pos = request.json['start']; N = 8
    moves_x = [1, 1, 2, 2, -1, -1, -2, -2]
    moves_y = [2, -2, 1, -1, 2, -2, 1, -1]

    def get_degree(board, r, c):
        return sum(1 for i in range(8)
                   if 0 <= c + moves_x[i] < N and 0 <= r + moves_y[i] < N
                   and board[r + moves_y[i]][c + moves_x[i]] == -1)

    board = [[-1] * N for _ in range(N)]
    curr_x = start_pos['x']; curr_y = start_pos['y']
    board[curr_y][curr_x] = 1
    path = [{'x': curr_x, 'y': curr_y}]

    for step in range(2, N * N + 1):
        min_deg = 9; next_x, next_y = -1, -1
        start_idx = random.randint(0, 7)
        for i in range(8):
            idx = (start_idx + i) % 8
            nx = curr_x + moves_x[idx]; ny = curr_y + moves_y[idx]
            if 0 <= nx < N and 0 <= ny < N and board[ny][nx] == -1:
                deg = get_degree(board, ny, nx)
                if deg < min_deg:
                    min_deg = deg; next_x, next_y = nx, ny
        if next_x == -1: break
        curr_x, curr_y = next_x, next_y
        board[curr_y][curr_x] = step
        path.append({'x': curr_x, 'y': curr_y})

    return jsonify({'path': path})


# ==========================================
# GAME 26: NEON CLUSTER (K-MEANS)
# ==========================================
@app.route('/solve_kmeans', methods=['POST'])
def solve_kmeans():
    data = request.json; points = data['points']; k = data.get('k', 3)
    if len(points) < k:
        return jsonify({'error': 'Not enough data points'})
    centroids = random.sample(points, k)
    assignments = [-1] * len(points)
    for _ in range(10):
        clusters = [[] for _ in range(k)]
        for i, p in enumerate(points):
            best_c = min(range(k), key=lambda c: (p['x']-centroids[c]['x'])**2 + (p['y']-centroids[c]['y'])**2)
            assignments[i] = best_c; clusters[best_c].append(p)
        centroids = [
            {'x': sum(p['x'] for p in cl) / len(cl), 'y': sum(p['y'] for p in cl) / len(cl)}
            if cl else centroids[c]
            for c, cl in enumerate(clusters)
        ]
    return jsonify({'centroids': centroids, 'assignments': assignments})


# ==========================================
# GAME 27: NEON LIFE (CONWAY'S GAME OF LIFE)
# ==========================================
@app.route('/evolve_life', methods=['POST'])
def evolve_life():
    grid = request.json['grid']; rows = len(grid); cols = len(grid[0])
    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    new_grid = [[0]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            live_neighbors = sum(
                1 for dr, dc in dirs
                if 0 <= r+dr < rows and 0 <= c+dc < cols and grid[r+dr][c+dc] == 1
            )
            if grid[r][c] == 1:
                new_grid[r][c] = 1 if 2 <= live_neighbors <= 3 else 0
            else:
                new_grid[r][c] = 1 if live_neighbors == 3 else 0
    return jsonify({'grid': new_grid})


# ==========================================
# GAME 28: NEON EVO (GENETIC ALGORITHM)
# ==========================================
@app.route('/solve_evo', methods=['POST'])
def solve_evo():
    data = request.json; target = data['target']; population = data['population']
    mutation_rate = 0.05

    def fitness(individual):
        return sum(1 for i, char in enumerate(individual) if i < len(target) and char == target[i])

    population.sort(key=lambda x: fitness(x), reverse=True)
    cutoff = max(1, int(len(population) * 0.2))
    next_gen = population[:cutoff]
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "

    while len(next_gen) < len(population):
        top_half = population[:max(1, int(len(population) * 0.5))]
        parent1 = random.choice(top_half); parent2 = random.choice(top_half)
        split = random.randint(0, len(target) - 1)
        child = list(parent1[:split] + parent2[split:])
        child = [random.choice(chars) if random.random() < mutation_rate else ch for ch in child]
        next_gen.append("".join(child))

    best_match = next_gen[0]; best_score = fitness(best_match)
    return jsonify({'population': next_gen, 'best': best_match,
                    'fitness': (best_score / len(target)) * 100})


# ==========================================
# GAME 29: NEON REGRESS (LINEAR REGRESSION)
# ==========================================
@app.route('/solve_regress', methods=['POST'])
def solve_regress():
    points = request.json['points']
    if len(points) < 2: return jsonify({'error': 'Need at least 2 data points.'})
    xs = [p['x'] for p in points]; ys = [p['y'] for p in points]; n = len(points)
    sum_x = sum(xs); sum_y = sum(ys)
    sum_xy = sum(x*y for x, y in zip(xs, ys))
    sum_xx = sum(x*x for x in xs)
    denom = n * sum_xx - sum_x ** 2
    if denom == 0: return jsonify({'error': 'Vertical line detected (Undefined slope).'})
    m = (n * sum_xy - sum_x * sum_y) / denom
    b = (sum_y - m * sum_x) / n
    return jsonify({
        'line': {'x1': 0, 'y1': b, 'x2': 800, 'y2': m * 800 + b},
        'equation': f"y = {m:.2f}x + {b:.2f}"
    })


# ==========================================
# GAME 30: NEON SORT (ALGORITHM VISUALIZER)
# ==========================================
@app.route('/solve_sort', methods=['POST'])
def solve_sort():
    data = request.json['array']; algo = request.json['algo']; steps = []
    if algo == 'bubble':
        arr = list(data); n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                steps.append({'type': 'compare', 'indices': [j, j+1]})
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
                    steps.append({'type': 'swap', 'indices': [j, j+1], 'values': [arr[j], arr[j+1]]})
    elif algo == 'quick':
        arr = list(data)
        def partition(low, high):
            pivot = arr[high]; i = low - 1
            steps.append({'type': 'pivot', 'indices': [high]})
            for j in range(low, high):
                steps.append({'type': 'compare', 'indices': [j, high]})
                if arr[j] <= pivot:
                    i += 1; arr[i], arr[j] = arr[j], arr[i]
                    steps.append({'type': 'swap', 'indices': [i, j], 'values': [arr[i], arr[j]]})
            arr[i+1], arr[high] = arr[high], arr[i+1]
            steps.append({'type': 'swap', 'indices': [i+1, high], 'values': [arr[i+1], arr[high]]})
            return i + 1
        def quickSort(low, high):
            if low < high:
                pi = partition(low, high)
                quickSort(low, pi - 1); quickSort(pi + 1, high)
        if len(arr) > 1: quickSort(0, len(arr) - 1)
    return jsonify({'steps': steps})


# ==========================================
# GAME 31: NEON PACK (BIN PACKING AI)
# ==========================================
@app.route('/solve_pack', methods=['POST'])
def solve_pack():
    items = request.json['items']; bin_capacity = 100
    items_sorted = sorted(items, reverse=True); bins = []
    for item in items_sorted:
        placed = False
        for b in bins:
            if b['space_left'] >= item:
                b['space_left'] -= item; b['items'].append(item)
                placed = True; break
        if not placed:
            bins.append({'space_left': bin_capacity - item, 'items': [item]})
    return jsonify({'bins': bins, 'items_sorted': items_sorted})


# ==========================================
# GAME 32: NEON HULL (CONVEX HULL ALGO)
# ==========================================
@app.route('/solve_hull', methods=['POST'])
def solve_hull():
    points = request.json['points']
    if len(points) < 3: return jsonify({'hull': points})
    def cross(o, a, b):
        return (a['x']-o['x'])*(b['y']-o['y']) - (a['y']-o['y'])*(b['x']-o['x'])
    points.sort(key=lambda p: (p['x'], p['y']))
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0: lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0: upper.pop()
        upper.append(p)
    return jsonify({'hull': lower[:-1] + upper[:-1]})


# ==========================================
# GAME 33: NEON SPAN (MINIMUM SPANNING TREE)
# ==========================================
@app.route('/solve_span', methods=['POST'])
def solve_span():
    points = request.json['points']; n = len(points)
    if n < 2: return jsonify({'edges': []})
    def dist(i, j):
        return math.hypot(points[i]['x']-points[j]['x'], points[i]['y']-points[j]['y'])
    key = [float('inf')] * n; parent = [-1] * n; visited = [False] * n
    key[0] = 0
    for _ in range(n):
        u = min((i for i in range(n) if not visited[i]), key=lambda i: key[i], default=-1)
        if u == -1: break
        visited[u] = True
        for v in range(n):
            if not visited[v]:
                w = dist(u, v)
                if w < key[v]: key[v] = w; parent[v] = u
    edges = [{'source': parent[i], 'target': i} for i in range(1, n) if parent[i] != -1]
    return jsonify({'edges': edges})


# ==========================================
# GAME 34: NEON FLOW (MAX FLOW ALGO)
# ==========================================
@app.route('/solve_flow', methods=['POST'])
def solve_flow():
    data = request.json; nodes = data['nodes']; edges = data['edges']
    source = 0; sink = len(nodes) - 1; n = len(nodes)

    capacity = [[0] * n for _ in range(n)]
    graph = [set() for _ in range(n)]
    for e in edges:
        u, v, cap = e['source'], e['target'], e['capacity']
        graph[u].add(v); graph[v].add(u)
        capacity[u][v] += cap

    max_flow = 0; paths = []

    while True:
        parent = [-1] * n; parent[source] = source
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in graph[u]:
                if parent[v] == -1 and capacity[u][v] > 0:
                    parent[v] = u
                    queue.append(v)

        if parent[sink] == -1:
            break

        path_flow = float('inf'); v = sink; current_path = []
        while v != source:
            u = parent[v]; path_flow = min(path_flow, capacity[u][v])
            current_path.append(v); v = u
        current_path.append(source); current_path.reverse()

        v = sink
        while v != source:
            u = parent[v]
            capacity[u][v] -= path_flow
            capacity[v][u] += path_flow
            v = u

        max_flow += path_flow
        paths.append({'path': current_path, 'flow': path_flow})

    return jsonify({'max_flow': max_flow, 'steps': paths})


# ==========================================
# GAME 35: NEON COLOR (GRAPH COLORING)
# ==========================================
@app.route('/solve_color', methods=['POST'])
def solve_color():
    data = request.json; nodes = data['nodes']; edges = data['edges']
    adj = {n['id']: [] for n in nodes}
    for e in edges:
        adj[e['source']].append(e['target'])
        adj[e['target']].append(e['source'])
    colors = {}
    sorted_nodes = sorted(nodes, key=lambda x: len(adj[x['id']]), reverse=True)
    for node in sorted_nodes:
        nid = node['id']
        neighbor_colors = {colors[nb] for nb in adj[nid] if nb in colors}
        color = 0
        while color in neighbor_colors: color += 1
        colors[nid] = color
    return jsonify({'colors': colors, 'count': max(colors.values()) + 1 if colors else 0})


# ==========================================
# GAME 36: NEON QUEEN (N-QUEENS BACKTRACKING)
# ==========================================
@app.route('/solve_queen', methods=['POST'])
def solve_queen():
    N = 8; board = [[0]*N for _ in range(N)]; steps = []
    def is_safe(board, row, col):
        for i in range(row):
            if board[i][col] == 1: return False
        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if board[i][j] == 1: return False
        for i, j in zip(range(row, -1, -1), range(col, N, 1)):
            if board[i][j] == 1: return False
        return True
    def solve(row):
        if row >= N: return True
        for col in range(N):
            if is_safe(board, row, col):
                board[row][col] = 1
                steps.append({'row': row, 'col': col, 'action': 'place'})
                if solve(row + 1): return True
                board[row][col] = 0
                steps.append({'row': row, 'col': col, 'action': 'remove'})
        return False
    solve(0)
    return jsonify({'steps': steps})


# ==========================================
# NEURAL LAB TOOLS
# ==========================================

@app.route('/tool_sentiment', methods=['POST'])
def tool_sentiment():
    try:
        data = request.get_json(); text = data.get('text', '')
        if not text: return jsonify({'html': '<span style="color:orange">No input data.</span>'})
        blob = TextBlob(text); score = blob.sentiment.polarity
        if score > 0.1:   mood, color = "POSITIVE", "#00ff00"
        elif score < -0.1: mood, color = "NEGATIVE", "#ff0055"
        else:              mood, color = "NEUTRAL",  "#ffff00"
        return jsonify({'html': f'<strong style="color:{color}">{mood}</strong> (Score: {round(score, 2)})'})
    except Exception as e:
        return jsonify({'html': f'<span style="color:red">ERROR: {str(e)}</span>'})

@app.route('/tool_summary', methods=['POST'])
def tool_summary():
    try:
        data = request.get_json(); text = data.get('text', '')
        if not text: return jsonify({'html': '<span style="color:red">No text provided.</span>'})
        blob = TextBlob(text)
        sentences = [str(s).strip() for s in blob.sentences if str(s).strip()]
        if len(sentences) > 2:
            summary = sentences[0] + "... " + sentences[-1] + "."
        else:
            summary = text
        return jsonify({'html': f'<strong>SUMMARY:</strong><br>{summary}'})
    except Exception as e:
        return jsonify({'html': f'<span style="color:red">Processing Failed: {str(e)}</span>'})

@app.route('/tool_spam', methods=['POST'])
def tool_spam():
    try:
        data = request.get_json(); text = data.get('text', '').lower()
        spam_words = ['free', 'winner', 'click here', 'urgent', 'buy now', 'cash', 'lottery', 'prize']
        detected = [w for w in spam_words if w in text]
        if detected:
            return jsonify({'html': f'<strong style="color:#ff0055">THREAT DETECTED</strong><br>Triggers: {", ".join(detected)}'})
        return jsonify({'html': '<strong style="color:#00ff00">CLEAN MESSAGE</strong><br>No standard threats found.'})
    except Exception as e:
        return jsonify({'html': f'<span style="color:red">Scan Error: {str(e)}</span>'})


# ── AUTH ROUTES ──────────────────────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password_hash, request.form['password']):
            login_user(user)
            return redirect('/')
        return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        if len(username) < 3:
            return render_template('register.html', error='Username must be at least 3 characters')
        if len(password) < 6:
            return render_template('register.html', error='Password must be at least 6 characters')
        if User.query.filter_by(username=username).first():
            return render_template('register.html', error='Username already taken')
        user = User(username=username, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect('/')
    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/')


@app.route('/save_score', methods=['POST'])
@login_required
def save_score():
    data = request.json
    game  = data.get('game', '').strip()
    value = int(data.get('score', 0))
    if not game or value <= 0:
        return jsonify({'status': 'ignored'})

    score = Score(user_id=current_user.id, game=game, score=value)
    db.session.add(score)
    db.session.commit()

    BADGES = {
        'FIRST_WIN':   lambda s: len(s) >= 1,
        'DECADE':      lambda s: len(s) >= 10,
        'CENTURION':   lambda s: len(s) >= 100,
        'MULTITASKER': lambda s: len(set(x.game for x in s)) >= 5,
        'GRANDMASTER': lambda s: len(set(x.game for x in s)) >= 20,
    }
    all_scores = Score.query.filter_by(user_id=current_user.id).all()
    earned     = {a.badge for a in Achievement.query.filter_by(user_id=current_user.id).all()}
    new_badges = []
    for badge, condition in BADGES.items():
        if badge not in earned and condition(all_scores):
            db.session.add(Achievement(user_id=current_user.id, badge=badge))
            new_badges.append(badge)
    if new_badges:
        db.session.commit()

    return jsonify({'status': 'saved', 'new_badges': new_badges})


@app.route('/profile')
@login_required
def profile():
    scores = Score.query.filter_by(user_id=current_user.id)\
                        .order_by(Score.played_at.desc()).all()
    achievements = Achievement.query.filter_by(user_id=current_user.id).all()
    return render_template('profile.html', scores=scores,
                           user=current_user, achievements=achievements)


@app.route('/leaderboard_data')
def leaderboard_data():
    top = db.session.query(Score, User)\
                    .join(User)\
                    .order_by(Score.score.desc())\
                    .limit(20).all()
    return jsonify([{
        'rank':   i + 1,
        'player': u.username,
        'game':   s.game,
        'score':  s.score,
        'cat':    'ai',
        'diff':   'med'
    } for i, (s, u) in enumerate(top)])


# ── CREATE DB TABLES ON STARTUP ───────────────────────────────────────────────
with app.app_context():
    db.create_all()


if __name__ == '__main__':
    app.run(debug=True)
