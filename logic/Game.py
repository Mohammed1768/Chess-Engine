import chess
from Board import Board

class Game:
    def __init__(self, depth=4):
        self.board = Board()
        self.depth = depth

    def player_move(self):
        while True:
            try:
                move_str = input("Your move (uci): ")

                move = chess.Move.from_uci(move_str)

                if move not in self.board.board.legal_moves:
                    print("Illegal move.")
                    continue

                self.board.move(move)
                break

            except Exception:
                print("Invalid move format.")

    def engine_move(self):
        print("Engine thinking...")

        move = self.board.get_best_move(self.depth)

        print(f"Engine plays: {move}")

        self.board.move(move)

    def play(self):
        print(self.board)
        print()

        while not self.board.board.is_game_over():

            # Human plays White
            if self.board.board.turn == chess.WHITE:
                self.player_move()

            # Engine plays Black
            else:
                self.engine_move()

            print()
            print(self.board)
            print()

        print("Game Over")

        if self.board.board.is_checkmate():
            winner = "Black" if self.board.board.turn == chess.WHITE else "White"
            print(f"{winner} wins by checkmate.")

        else:
            print("Draw.")


game = Game(depth=4)
game.play()