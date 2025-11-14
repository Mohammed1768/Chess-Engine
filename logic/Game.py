import chess
from Board import Board

class Game:
    def __init__(self, depth=5):
        self.board = Board()
        self.k = depth
        self.positions = {}

    def get_best_move(self):
        best_move, best_eval = self.minimax(self.k, -9999999, 9999999)
        return best_move

    def minimax(self, k, alpha, beta):
        if self.board.is_game_over():
            if self.board.is_checkmate():
                if self.board.turn == chess.WHITE: return None, -9999
                else: return None, 9999
            return None, 0


        if k == 0:
            return None, self.evaluate()

        best_move = None

        if self.board.turn == chess.WHITE:
            max_eval = -99999
            for move in list(self.board.legal_moves):
                self.board.push(move)
                _, eval_score = self.minimax(k-1, alpha, beta)
                self.board.pop()

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if alpha >= beta: break

            return best_move, max_eval

        min_eval = 99999
        for move in list(self.board.legal_moves):
            self.board.push(move)
            _, eval_score = self.minimax(k-1, alpha, beta)
            self.board.pop()

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move

            beta = min(beta, eval_score)
            if beta<=alpha: break
        return best_move, min_eval
    

    def move(self, move: chess.Move):
        print(f"{"White: "if self.board.turn==chess.WHITE else "Black: "}", move)
        self.board.push(move)
    

    def evaluate(self):
        fen = self.board.fen()
        if fen not in self.positions:
            self.positions[fen] = float(self.board.get_eval())
        return self.positions[fen]

game = Game(4)
for _ in range(30):
    m = game.get_best_move()
    game.move(m)
