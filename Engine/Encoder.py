import numpy as np

class Encoder:
    """
    Takes in a FEN string and returns a (18, 8, 8) numpy array to feed to the CNN
    Example: r3k2r/1b2bppp/p1n1pn2/1p2N1B1/8/2NB1P2/PPP3PP/2KR3R w kq - 1 14
    """

    def encode_FEN(self, fen: str, turn: str, castling: str, en_passant: str):
        piece_map = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}
        rows = fen.split('/')

        # 12 planes for pieces
        board = np.zeros((12, 8, 8), dtype=np.float32)

        for r, row in enumerate(rows):
            c = 0
            for ch in row:
                if ch.isdigit():
                    c += int(ch)
                else:
                    plane = piece_map[ch.lower()]
                    if ch.islower():  # black piece
                        plane += 6
                    board[plane, 7 - r, c] = 1  # flip vertically so rank 8 is row 0
                    c += 1

        # 4 planes for castling rights
        castling_planes = np.zeros((4, 8, 8), dtype=np.float32)
        if 'K' in castling: castling_planes[0] = 1
        if 'Q' in castling: castling_planes[1] = 1
        if 'k' in castling: castling_planes[2] = -1
        if 'q' in castling: castling_planes[3] = -1

        # 1 plane for turn
        turn_plane = np.ones((1, 8, 8), dtype=np.float32)
        if turn == 'b':
            turn_plane *= -1

        # 1 plane for en passant square
        enpassant_plane = np.zeros((1, 8, 8), dtype=np.float32)
        if en_passant != '-':
            file = "abcdefgh".index(en_passant[0])
            rank = 8 - int(en_passant[1])
            enpassant_plane[0, rank, file] = 1 if turn == 'w' else -1

        # Combine all planes -> shape (18, 8, 8)
        planes = np.concatenate([board, turn_plane, castling_planes, enpassant_plane], axis=0)
        return planes
