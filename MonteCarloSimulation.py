import random
from core import *
from chess import *
from numpy.random import dirichlet

class Node:

    def __init__(self, state: Board, value: float=0, transitions : {str: ()} = {}, total = 0):
        #transitions : {str -> action: ((N, Q, P), Node-> new state)}
        self.state = state
        self.value = value
        self.transitions = transitions
        self.total = total

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
        c = 0
        for i in range(64):
            for j in range(64):
                if i != j:
                    self.all_moves_ids[Move(i, j)] = c
                    self.ids_all_moves[c] = Move(i, j)
                    c += 1


    def runSimAndReturnBest(self, states: [BoardInformation], iterations: int, actions, model, player, tau):
        self.actions = set(actions)
        children = []
        distribution = []
        for i, each in enumerate(states):
            children.append(Node(each.board_state.copy(stack=False)))
            distribution.append(each.p)
        sampler = SamplerRoot(children, distribution)
        self.visited = set()
        for _ in range(iterations):
            root = random.choices(sampler.children, weights=sampler.distribution)[0]
            root.state.turn = player
            root, _ = self.runOneAndUpdate(root, player, 1, self.actions.intersection(root.state.legal_moves), model)

        all_actions_distribution = defaultdict(float)
        # all_actions_rewards = defaultdict(float)
        total = 0
        # action_distribution = defaultdict(list)

        for root in sampler.children:
            for key, value in root.transitions.items():
                if key in self.actions:
                    all_actions_distribution[key] += value[0][0]#math.pow(value[0][0], 1/tau)
                    # values = action_distribution.get(key, [0, 0, 0])
                    # action_distribution[key] = [values[0]+value[0][0], values[1]+value[0][1], values[2]+value[0][2]]
                    total += value[0][0]#math.pow(value[0][0], 1/tau)
        # print(all_actions_rewards, all_actions_distribution)
        if total != 0:
            for each in all_actions_distribution:
                all_actions_distribution[each] /= total
        new_distribution = [0]*len(self.all_moves_ids)
        for move, value in all_actions_distribution.items():
            id_ = self.all_moves_ids.get(move)
            if id_ is None:
                move = Move(each.from_square, each.to_square)
                id_ = self.all_moves_ids.get(move)
            new_distribution[id_] = value
        # for each in all_actions_rewards:
        #     all_actions_rewards[each] /= total
        actionId = random.choice(self.allmax(list(all_actions_distribution.values())))
        action = list(all_actions_distribution.keys())[actionId]
        return action, new_distribution #, value, prior_distribution

    def runOneAndUpdate(self, root, player_color, discount, actions, model):
        root.state.turn = player_color
        state_value = root.state.is_game_over()
        if state_value == True:
            win = False
            winner = root.state.result().split("-")[0]
            if winner == "1":
                win = True
            if win == player_color:
                root.val = 1
            elif winner == "1/2":
                root.val = 0
            elif win != player_color:
                root.val = -1
            return root, -root.value
        else:
            state_actions = actions
            action_info = []
            priorsTotal = 0
            root.state.turn = player_color
            state_fen = self.stringify(root.state.piece_map())
            if state_fen not in self.visited:
                self.visited.add(state_fen)
                input = self.core.OHVETransform(root.state, player_color)
                inp = np.array([input])
                out, val = model(inp, training=False)

                root.value = val.numpy()[0]
                out = out.numpy()[0]
                # print(state_actions)
                action_space = 0
                for each in state_actions:
                    action_space += 1
                    id_ = self.all_moves_ids.get(each)
                    # print(id_)
                    if id_ is None:
                        # print(each)
                        id_ = Move(from_square=each.from_square, to_square=each.to_square)
                        id_ = self.all_moves_ids.get(id_)
                    prior = out[id_]
                    root.state.turn = player_color
                    root.transitions[each] = [[0, 0, prior], Node(root.state.copy(stack=False), transitions={})]
                    # print("PRIOR: ", root.transitions[each], each)
                    priorsTotal += prior
                dirichlet_noise = dirichlet([0.3]*action_space)
                # Normalizing priors and adding dirichlet noise
                for i, each in enumerate(state_actions):
                    # if each not in root.transitions:
                    #     each = Move(from_square=each.from_square, to_square=each.to_square)
                    p = root.transitions[each][0][2]/priorsTotal
                    root.transitions[each][0][2] = (1-self.epsilon)*p + (self.epsilon*dirichlet_noise[i])
                return root, -root.value
                # print(player_color, "ACTIONS:", state_actions, root.transitions)
                # print(priorsTotal)

            for each in state_actions:
                if each not in root.transitions:
                    root.transitions[each] = [[0, 0, 0], Node(root.state.copy(stack=False))]
                else:
                    root.transitions[each][1].state = root.state.copy(stack=False)
                action_info.append(root.transitions.get(each)[0])
            # print(action_info, "--", root.transitions)
            # if len(action_info) == 0:
            #     print("ERROR NO ACTIONS FOUND")
            #     return root, -root.value
            actionId = self.pickAction(action_info, root.total)
            action = list(state_actions)[actionId]
            #print(root.transitions[action][1].state.fen(), root.state.fen())
            root.total += 1
            # if action not in root.state.legal_moves:
            #     return root, -root.value
            # if action not in root.transitions:
            #     action = Move(from_square=action.from_square, to_square=action.to_square)
            # if action not in root.transitions:
            #     return root, -root.val
            root.transitions[action][1].state.turn = player_color
            root.transitions[action][1].state.push(action)
            # root.transitions[action][1].state.turn = not player_color
            # print(player_color, root.transitions[action][1].transitions)
            root.transitions[action][1], val = self.runOneAndUpdate(root.transitions[action][1], not player_color, discount, root.transitions[action][1].state.legal_moves, model)
            root.transitions[action][0][1] += val
            root.value += val
            root.transitions[action][0][0] += 1
            return root, -val

    def pickAction(self, action_values, total, c=1000):
        Nc = (total**0.5)*c
        ucb_values = np.array([q+(p*Nc/(1+a_t)) for a_t, q, p in action_values])
        best_vals = self.allmax(ucb_values)
        if len(best_vals) > 1:
            return random.choice(best_vals)
        else:
            return best_vals[0]
    # @profile
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

    def stringify(self, p):
        s = ""
        for i in range(64):
            if i in p:
                s += p[i].symbol()
            else:
                s += "."
        return s



