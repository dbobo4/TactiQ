import numpy as np
from tictactoeboard import TicTacToeBoard


class TicTacToeGame:
    """
    AlphaZero-szerű játék-API 4x4-es, 3 hosszú nyeréshez.

    A state mindenhol egy (4, 4) alakú numpy tömb:
        0  : üres mező
        +1 : aktuális játékos
        -1 : ellenfél

    self.change_perspective(state, player): megszorozza a táblát player-rel,
    így a soron következő játékos mindig +1-esként látja magát.
    """

    def __init__(self, size: int = 4, win_length: int = 3) -> None:
        self.board_helper = TicTacToeBoard(size=size, win_length=win_length)
        self.row_count = size
        self.column_count = size
        self.action_size = size * size

    # --------- alap műveletek ---------

    def get_initial_state(self) -> np.ndarray:
        return np.zeros((self.row_count, self.column_count), dtype=int)

    def get_next_state(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        """
        Visszaadja az új állapotot, miután 'player' lerakta a jelölését 'action' pozícióra.
        Nem módosítja az eredeti state-et, új másolatot ad vissza.
        """
        next_state = state.copy()
        row = action // self.column_count
        col = action % self.column_count
        next_state[row, col] = player
        return next_state

    def get_valid_moves(self, state: np.ndarray) -> np.ndarray:
        """
        Visszaad egy (action_size,) alakú 0/1 maszkot, ahol 1 a szabad mező.
        """
        return (state.reshape(-1) == 0).astype(np.uint8)

    # --------- terminális állapotérték ---------

    def get_value_and_terminated(self, state: np.ndarray, action: int):
        """
        Értékeli az állapotot és jelzi, hogy terminális-e.

        Visszatérés:
            value: 1  → az a játékos nyert, aki az utolsó lépést tette
                    0  → döntetlen vagy még nem ért véget a játék
            terminated: True, ha a játék véget ért (nyerés vagy döntetlen)
        """
        if action is None:
            # gyökérnél nincs előző lépés
            return 0, False

        row = action // self.column_count
        col = action % self.column_count
        last_player = state[row, col]  # +1 vagy -1

        if last_player == 0:
            # elvileg nem fordulhat elő, de legyen védőháló
            return 0, False

        # betöltjük a state-et a helper boardba, hogy a már DQN-ben használt win logikát használjuk újra
        self.board_helper.board[:, :] = state

        if self.board_helper.check_is_win(last_player):
            return 1, True

        if self.board_helper.check_is_draw():
            return 0, True

        return 0, False

    # --------- perspektíva váltás ---------

    @staticmethod
    def get_opponent(player: int) -> int:
        return -player

    @staticmethod
    def get_opponent_value(value: float) -> float:
        return -value

    @staticmethod
    def change_perspective(state: np.ndarray, player: int) -> np.ndarray:
        """
        Perspektívaváltás: a megadott 'player' nézőpontjából nézzük a táblát.
        """
        return state * player

    # Example encoding for a 2x2 board:
    #   state =
    #       [[ 1,  0],
    #        [-1,  1]]
    #   encoded_state has shape (3, 2, 2):
    #       channel 0 (enemy, state == -1):
    #           [[0., 0.],
    #            [1., 0.]]
    #       channel 1 (empty, state == 0):
    #           [[0., 1.],
    #            [0., 0.]]
    #       channel 2 (own, state == 1):
    #           [[1., 0.],
    #            [0., 1.]]
    @staticmethod
    def get_encoded_state(state: np.ndarray) -> np.ndarray:
        """
        3 csatornás one-hot kódolás:
        [ellenfél kövei, üres mezők, saját kövek]  → alak: (3, size, size)
        """
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        return encoded_state
