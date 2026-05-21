import chess
import chess.polyglot
import torch
from pathlib import Path
from Engine.ChessNet import ChessNet
from dataset.Encoder import Encoder


class Board:
    def __init__(self, model_path=Path(__file__).resolve().parents[1] / "chessnet_weights.pth"):
        self.board = chess.Board()

        self.device = torch.device("cpu")
        torch.set_num_threads(4)

        model = ChessNet().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()

        self.model = model

        self.transposition_table = {}

    def evaluate_board(self, depth):
        if self.board.is_checkmate():
            return -9999 - depth if self.board.turn == chess.WHITE else 9999 + depth
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0

        x = Encoder.encode_FEN(self.board.fen())
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            return self.model(x).item()

    def order_moves(self, legal_moves, hash_move=None):
        def score_move(move):
            if move == hash_move:
                return 10000

            score = 0

            if self.board.is_capture(move):
                victim = self.board.piece_at(move.to_square)
                attacker = self.board.piece_at(move.from_square)

                victim_val = victim.piece_type if victim else 1
                attacker_val = attacker.piece_type if attacker else 1

                score += 1000 + (victim_val * 10) - attacker_val

            if move.promotion:
                score += 900

            if self.board.gives_check(move):
                score += 500

            return score

        return sorted(legal_moves, key=score_move, reverse=True)

    def alpha_beta(self, depth, alpha, beta):
        alpha_orig = alpha
        beta_orig = beta

        board_hash = chess.polyglot.zobrist_hash(self.board)

        if board_hash in self.transposition_table:
            tt_entry = self.transposition_table[board_hash]

            if tt_entry["depth"] >= depth:
                if tt_entry["type"] == "EXACT":
                    return tt_entry["value"]

                elif tt_entry["type"] == "LOWERBOUND":
                    alpha = max(alpha, tt_entry["value"])

                elif tt_entry["type"] == "UPPERBOUND":
                    beta = min(beta, tt_entry["value"])

                if alpha >= beta:
                    return tt_entry["value"]

        if depth == 0 or self.board.is_game_over():
            return self.evaluate_board(depth)

        hash_move = self.transposition_table.get(board_hash, {}).get("best_move", None)
        ordered_moves = self.order_moves(self.board.legal_moves, hash_move)

        best_move_this_node = None

        if self.board.turn == chess.WHITE:
            best_eval = float("-inf")

            for move in ordered_moves:
                self.board.push(move)
                evaluation = self.alpha_beta(depth - 1, alpha, beta)
                self.board.pop()

                if evaluation > best_eval:
                    best_eval = evaluation
                    best_move_this_node = move

                alpha = max(alpha, evaluation)

                if alpha >= beta:
                    break

        else:
            best_eval = float("inf")

            for move in ordered_moves:
                self.board.push(move)
                evaluation = self.alpha_beta(depth - 1, alpha, beta)
                self.board.pop()

                if evaluation < best_eval:
                    best_eval = evaluation
                    best_move_this_node = move

                beta = min(beta, evaluation)

                if alpha >= beta:
                    break

        if best_eval <= alpha_orig:
            tt_type = "UPPERBOUND"
        elif best_eval >= beta_orig:
            tt_type = "LOWERBOUND"
        else:
            tt_type = "EXACT"

        self.transposition_table[board_hash] = {
            "depth": depth,
            "value": best_eval,
            "type": tt_type,
            "best_move": best_move_this_node,
        }

        return best_eval

    def get_best_move(self, k=3):
        best_move = None

        for current_depth in range(1, k + 1):
            ordered_moves = self.order_moves(self.board.legal_moves, best_move)

            alpha = float("-inf")
            beta = float("inf")

            if self.board.turn == chess.WHITE:
                best_eval = float("-inf")

                for move in ordered_moves:
                    self.board.push(move)
                    score = self.alpha_beta(current_depth - 1, alpha, beta)
                    self.board.pop()

                    if score > best_eval:
                        best_eval = score
                        best_move = move

                    alpha = max(alpha, score)

            else:
                best_eval = float("inf")

                for move in ordered_moves:
                    self.board.push(move)
                    score = self.alpha_beta(current_depth - 1, alpha, beta)
                    self.board.pop()

                    if score < best_eval:
                        best_eval = score
                        best_move = move

                    beta = min(beta, score)

        return best_move

    def move(self, move):
        if isinstance(move, str):
            move = chess.Move.from_uci(move)

        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move}")

        self.board.push(move)

    def __str__(self):
        return str(self.board)
