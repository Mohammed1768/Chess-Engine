import numpy as np

class Encoder:
    _buffer = np.zeros((18, 8, 8), dtype=np.float32)
    
    _char_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,   # White pieces
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
    }
    _digit_map = {str(i): i for i in range(1, 9)} 

    @classmethod
    def encode_FEN(cls, string: str):
        cls._buffer.fill(0)
        
        parts = string.split(' ')
        fen = parts[0]
        turn = parts[1]
        castling = parts[2]
        en_passant = parts[3]

        r = 0
        c = 0
        for ch in fen:
            if ch == '/':
                r += 1
                c = 0
            elif ch in cls._digit_map:
                c += cls._digit_map[ch]
            else:
                plane = cls._char_map[ch]
                cls._buffer[plane, r, c] = 1
                c += 1

        if turn == 'b':
            cls._buffer[12].fill(1) 

        if castling != '-':
            if 'K' in castling: cls._buffer[13].fill(1)
            if 'Q' in castling: cls._buffer[14].fill(1)
            if 'k' in castling: cls._buffer[15].fill(1)
            if 'q' in castling: cls._buffer[16].fill(1)

        if en_passant != '-':
            file_idx = ord(en_passant[0]) - ord('a') 
            rank_idx = 8 - int(en_passant[1])        
            cls._buffer[17, rank_idx, file_idx] = 1

        return cls._buffer.copy()