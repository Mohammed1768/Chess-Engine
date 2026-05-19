import numpy as np
from Encoder import Encoder

def parse_line(line: str):
    """
    Parses a line where FEN and decimal evaluation are split by a comma or tab.
    Example line: "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 , 0.35"
    """
    parts = line.strip().split(',')
    fen = parts[0].strip()
    evaluation = float(parts[1].strip())
        
    return fen, evaluation 

def parse_file(input_file : str, output_file : str):
    mats = []
    evals = []

    with open(input_file) as f:
        next(f)
        for idx, line in enumerate(f):
            fen, eval = parse_line(line)
            mat = Encoder.encode_FEN(fen)

            mats.append(mat)
            evals.append(eval)

    x = np.array(mats, dtype=np.uint8)
    y = np.array(evals, dtype=np.float32)

    np.savez(output_file, x=x, y=y)

    print(f"Successfully encoded {input_file} into {output_file}\n")

parse_file("train.csv", "train.npz")
parse_file("test.csv", "test.npz")
