import math
import random
from core import BoardInformation
from chess import *
from core import *
from numpy.random import dirichlet


class Node:
    def __init__(self, states: List[Board], chosenBoard: Board, value: float = 0, transitions: {str: ()} = {},
                 totals={}):
        # transitions : {str -> action: ((N, Q, P), Node-> new state)}
        self.states = states
        self.value = value
        self.transitions = transitions
        self.total = totals
        self.chosenBoard = chosenBoard


class SamplerRoot:
    def __init__(self, children, distribution):
        self.children = children
        self.distribution = distribution


class MonteCarlo:
    def __init__(self):
        self.actions = None
        self.core = Core(None, None)
        self.all_moves_ids = defaultdict(int)
        self.ids_all_moves = defaultdict(Move)
        self.visited = set()
        self.epsilon = 0.25
        k = 0
        for i in range(64):
            for j in range(64):
                if i != j:
                    self.all_moves_ids[Move(i, j)] = k
                    self.ids_all_moves[k] = Move(i, j)
                    k += 1

    @profile
    def runSimAndReturnBest(self, states: [BoardInformation], iterations: int, actions, model, player, tau, c,
                            opponent=False, training=False):
        self.isTraining = training
        self.c = c
        self.actions = set(actions)
        self.visited = {}
        allBoards = []
        for i, each in enumerate(states):
            allBoards.append(each.board_state)
            board_string = self.stringify(each.board_state)
            self.visited[board_string] = set()

        st = time.time()
        et = time.time()
        i = 0
        parentNode = Node(allBoards, None)
        while i < iterations:
            root = random.choice(allBoards)
            root.turn = player
            board_string = self.stringify(root)
            i += 1
            parentNode.chosenBoard = root.copy(stack=False)
            parentNode, _ = self.runOneAndUpdate(parentNode, player, 1,
                                                 self.actions.intersection(parentNode.chosenBoard.pseudo_legal_moves),
                                                 model, board_string)
            et = time.time()
        print(et - st)

        all_actions_distribution = defaultdict(float)
        total = 0
        # print(parentNode.transitions)
        for key, value in parentNode.transitions.items():
            if key in self.actions:
                all_actions_distribution[key] += value[0][0]  # math.pow(value[0][0], 1/tau)
                total += value[0][0]
        if total != 0:
            for each in all_actions_distribution:
                all_actions_distribution[each] /= total
        new_distribution = [0] * len(self.all_moves_ids)
        for move, value in all_actions_distribution.items():
            id_ = self.all_moves_ids.get(move)
            if id_ is None:
                move = Move(each.from_square, each.to_square)
                id_ = self.all_moves_ids.get(move)
            new_distribution[id_] = value
        actionId = random.choice(self.allmax(list(all_actions_distribution.values())))
        action = list(all_actions_distribution.keys())[actionId]
        return action, new_distribution

    @profile
    def runOneAndUpdate(self, root, player_color, discount, actions, model, board_string):
        root.chosenBoard.turn = player_color
        state_value = self.winner(root.states, player_color)
        if state_value:
            winner = state_value.split("-")[0]
            if winner == "1/2":
                root.val = 0
                return root, 0
            if winner == "1" and player_color:
                root.val = 1
                return root, -1
            if winner == "1" and not player_color:
                root.val = -1
                return root, 1
            if winner == "0" and not player_color:
                root.val = 1
                return root, -1
            if winner == "0" and player_color:
                root.val = -1
                return root, 1
        else:
            state_actions = actions
            action_info = []

            root.chosenBoard.turn = player_color
            state_fen = self.stringify(root.chosenBoard)
            if state_fen not in self.visited[board_string]:
                priorsTotal = 0
                self.visited[board_string].add(state_fen)
                input = self.core.OHVETransform(root.chosenBoard, player_color)
                inp = np.array([input])
                out, val = model(inp, training=False)
                root.value += val.numpy()[0]
                out = out.numpy()[0]
                action_space = 0
                allStatesCopy = []
                for state in root.states:
                    allStatesCopy.append(state.copy(stack=False))
                root.chosenBoard.turn = player_color
                for each in state_actions:
                    action_space += 1
                    id_ = self.all_moves_ids.get(each)
                    if id_ is None:
                        id_ = Move(from_square=each.from_square, to_square=each.to_square)
                        id_ = self.all_moves_ids.get(id_)
                    prior = out[id_]
                    if each not in root.transitions:
                        root.transitions[each] = [[0, 0, prior], Node(allStatesCopy, root.chosenBoard.copy(stack=False),
                                                                      transitions={})]
                    else:
                        root.transitions[each][0][2] += prior

                for key in root.transitions:
                    priorsTotal += root.transitions[key][0][2]
                if self.isTraining:
                    dirichlet_noise = dirichlet([0.3] * action_space)
                    # Normalizing priors and adding dirichlet noise
                    for i, each in enumerate(state_actions):
                        p = root.transitions[each][0][2] / priorsTotal
                        root.transitions[each][0][2] = (1 - self.epsilon) * p + (self.epsilon * dirichlet_noise[i])
                else:
                    for i, each in enumerate(state_actions):
                        p = root.transitions[each][0][2] / priorsTotal
                        root.transitions[each][0][2] = p
                return root, -root.value

            if board_string not in root.total:
                root.total[board_string] = 0
            allStatesCopy = []
            for state in root.states:
                allStatesCopy.append(state.copy(stack=False))
            for each in state_actions:
                action_info.append(root.transitions.get(each, [[0, 0, 0]])[0])
            actionId = self.pickAction(action_info, root.total[board_string])
            action = list(state_actions)[actionId]
            if action not in root.transitions:
                root.transitions[action] = [[0, 0, 0],
                                            Node(allStatesCopy, root.chosenBoard.copy(stack=False), transitions={})]
            else:
                root.transitions[action][1].states = allStatesCopy
                root.transitions[action][1].chosenBoard = root.chosenBoard.copy(stack=False)
            root.total[board_string] += 1
            finalStates = []
            for board in root.transitions[action][1].states:
                try:
                    board.push(action)
                    finalStates.append(board)
                except:
                    continue

            root.transitions[action][1].states = finalStates

            root.transitions[action][1].chosenBoard.turn = player_color
            root.transitions[action][1].chosenBoard.push(action)
            root.transitions[action][1], val = self.runOneAndUpdate(root.transitions[action][1], not player_color,
                                                                    discount, root.transitions[action][
                                                                        1].chosenBoard.pseudo_legal_moves, model,
                                                                    board_string)
            root.transitions[action][0][1] += val
            root.value += val
            root.transitions[action][0][0] += 1
            return root, -val

    @profile
    def pickAction(self, action_values, total):
        K = math.sqrt(90)
        ucb_values = [self.uctval(a_t, q, K, total, p) for a_t, q, p in action_values]
        # ucb_values = np.array([ for a_t, q, p in action_values])
        best_vals = self.allmax(ucb_values)
        if len(best_vals) > 1:
            return random.choice(best_vals)
        else:
            return best_vals[0]

    # @profile
    @profile
    def uctval(self, a_t, q, c, t, p):
        return q + ((t * c) * (p / (1 + a_t)))

    @profile
    def allmax(self, inp):
        if len(inp) == 0:
            return []
        all_ = [0]
        max_ = inp[0]
        for i in range(1, len(inp)):
            if inp[i] > max_:
                all_ = [i]
                max_ = inp[i]
            elif inp[i] == max_:
                all_.append(i)
        return all_

    @profile
    def stringify(self, p):
        p = p.piece_map()
        s = ""
        for i in range(64):
            if i in p:
                s += p[i].symbol()
            else:
                s += "."
        return s

    @profile
    def isGameOver(self, board):
        p = board.piece_map()
        s = ""
        for i in range(64):
            if i in p:
                s += p[i].symbol()
            else:
                s += "."
        state_value = board.is_game_over()
        if state_value:
            if board.result() == "1/2-1/2":
                return True
        return not ("k" in s and "K" in s)

    @profile
    def winner(self, boards, player):
        for board in boards:
            # board.turn = player
            # s = str(board).replace(" ", "").replace("\n", "")
            # p = board.piece_map()
            # s = ""
            # for i in range(64):
            #     if i in p:
            #         s += p[i].symbol()
            #     else:
            #         s += "."

            if board.king(False) is None:
                return "1-0"
            elif board.king(True) is None:
                return "0-1"
            # else:
                # state_value = board.is_game_over()
                # if state_value:
                #     return board.result()
        return None
