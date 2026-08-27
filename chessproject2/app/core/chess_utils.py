import chess

def get_ascii_board(board: chess.Board) -> str:
    """Generates an ASCII board with rank/file coordinates using standard 1-byte characters."""
    lines = ["  +-----------------+"]
    for rank in range(7, -1, -1):
        row = [board.piece_at(chess.square(file, rank)) for file in range(8)]
        row_str = " ".join(p.symbol() if p else "." for p in row)
        lines.append(f"{rank + 1} | {row_str} |")
    lines.append("  +-----------------+")
    lines.append("    a b c d e f g h")
    return "\n".join(lines)