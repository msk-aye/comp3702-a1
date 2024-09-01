import sys
import time
import itertools

from game_env import GameEnv
from game_state import GameState
import transition as tr

"""
solution.py

This file is a template you should use to implement your solution.

You should implement each of the method stubs below. You may add additional methods and/or classes to this file if you 
wish. You may also create additional source files and import to this file if you wish.

COMP3702 Assignment 2 "Dragon Game" Support Code

Last updated by njc 30/08/23
"""


class Solver:

    def __init__(self, game_env: GameEnv):
        # TODO: Define any class instance variables you require (e.g. dictionary mapping state to VI value) here.

        self.game_env = game_env

    @staticmethod
    def testcases_to_attempt():
        """
        Return a list of testcase numbers you want your solution to be evaluated for.
        """
        return [1, 2, 3, 4, 5]

    # === Value Iteration ==============================================================================================

    def vi_initialise(self):
        """
        Initialise any variables required before the start of Value Iteration.
        """
        # TODO: Implement any initialisation for Value Iteration (e.g. building a list of states) here.
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.

        self.states = set(GameState(pos[0], pos[1], gem) for pos in self.possible_player_positions()
                          for gem in self.possible_gem_states())
        self.absorbing_state = GameState(-1, -1, (1,))
        self.states.add(self.absorbing_state)
        self.states_cache = dict()
        self.values = {state: 0.0 for state in self.states}
        self.policy = {state: self.game_env.WALK_RIGHT for state in self.states}
        self.converged = False


    def vi_is_converged(self):
        """
        Check if Value Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        # TODO: Implement code to check if Value Iteration has reached convergence here.
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.

        if self.max_diff_vi < self.game_env.epsilon:
            self.converged = True
            
        return self.converged

    def vi_iteration(self):
        """
        Perform a single iteration of Value Iteration (i.e. loop over the state space once).
        """
        # TODO: Implement code to perform a single iteration of Value Iteration here.
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.

        self.new_policy = dict()
        self.max_diff_vi = 0

        for state in self.states:

            best_value = -float("inf")
            best_action = None

            for action in GameEnv.ACTIONS:
                total = 0

                # Check if combination has appeared before and is stored in the cache
                if ((state, action) in self.states_cache.keys() and self.states_cache[(state, action)] != None):
                    for next_state, reward, prob in self.states_cache[(state, action)]:

                        if self.is_absorbing(next_state):
                            best_action = None
                            best_value = 0
                            break

                        total += prob * (reward + (self.game_env.gamma * self.values[next_state]))

                    if total > best_value:
                        best_value = total
                        best_action = action

                elif (self.game_env.perform_action(state, action, self.game_env.episode_seed)[0]):
                    self.states_cache[(state, action)] = tr.get_transition_outcomes(self.game_env, state, action)

                    for next_state, reward, prob in self.states_cache[(state, action)]:

                        if self.is_absorbing(next_state):
                            best_action = None
                            best_value = 0
                            break

                        total += prob * (reward + (self.game_env.gamma * self.values[next_state]))

                    if total > best_value:
                        best_value = total
                        best_action = action

            differences = abs(self.values[state] - best_value)
            if differences > self.max_diff_vi:
                self.max_diff_vi = differences

            self.values[state] = best_value
            self.new_policy[state] = best_action

        self.vi_is_converged()

        self.policy = self.new_policy

    def vi_plan_offline(self):
        """
        Plan using Value Iteration.
        """
        # In order to ensure compatibility with tester, you should not modify this method
        self.vi_initialise()
        while True:
            self.vi_iteration()

            # vi_iteration is always called before vi_is_converged
            if self.vi_is_converged():
                break

    def vi_get_state_value(self, state: GameState):
        """
        Retrieve V(s) for the given state.
        :param state: the current state
        :return: V(s)
        """
        # TODO: Implement code to return the value V(s) for the given state (based on your stored VI values) here. If a
        #  value for V(s) has not yet been computed, this function should return 0.
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.

        return self.values[state]


    def vi_select_action(self, state: GameState):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        # TODO: Implement code to return the optimal action for the given state (based on your stored VI values) here.
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.

        return self.policy[state]

    def print_policy(self):
        """
        Print state: policy action for every state in the state space.
        """
        for state, action in self.policy.items():
            print(state, action)

    # === Policy Iteration =============================================================================================

    def pi_initialise(self):
        """
        Initialise any variables required before the start of Policy Iteration.
        """
        # TODO: Implement any initialisation for Policy Iteration (e.g. building a list of states) here.
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.

        self.states = set(GameState(pos[0], pos[1], gem) for pos in self.possible_player_positions()
                          for gem in self.possible_gem_states())
        self.absorbing_state = GameState(-1, -1, (1,))
        self.states.add(self.absorbing_state)
        self.states_cache = dict()
        self.values = {state: 0.0 for state in self.states}
        self.policy = {state: self.game_env.WALK_RIGHT for state in self.states}
        self.converged = False

    def pi_is_converged(self):
        """
        Check if Policy Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        # TODO: Implement code to check if Policy Iteration has reached convergence here.
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.

        if self.max_diff_pi < self.game_env.epsilon:
            self.converged = True

        return self.converged

    def pi_iteration(self):
        """
        Perform a single iteration of Policy Iteration (i.e. perform one step of policy evaluation and one step of
        policy improvement).
        """
        # TODO: Implement code to perform a single iteration of Policy Iteration (evaluation + improvement) here.
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.

        self.new_policy = dict()
        self.max_diff_pi = 0

        for state in self.states:

            best_value = -float("inf")
            best_action = None

            for action in GameEnv.ACTIONS:
                total = 0

                # Check if combination has appeared before and is stored in the cache
                if ((state, action) in self.states_cache.keys() and self.states_cache[(state, action)] != None):
                    for next_state, reward, prob in self.states_cache[(state, action)]:

                        if self.is_absorbing(next_state):
                            best_action = None
                            best_value = 0
                            break

                        total += prob * \
                            (reward + (self.game_env.gamma *
                             self.values[next_state]))

                    if total > best_value:
                        best_value = total
                        best_action = action

                elif (self.game_env.perform_action(state, action, self.game_env.episode_seed)[0]):
                    self.states_cache[(state, action)] = tr.get_transition_outcomes(self.game_env, state, action)

                    for next_state, reward, prob in self.states_cache[(state, action)]:

                        if self.is_absorbing(next_state):
                            best_action = None
                            best_value = 0
                            break

                        total += prob * (reward + (self.game_env.gamma * self.values[next_state]))

                    if total > best_value:
                        best_value = total
                        best_action = action

            differences = abs(self.values[state] - best_value)
            if differences > self.max_diff_pi:
                self.max_diff_pi = differences

            self.values[state] = best_value
            self.new_policy[state] = best_action

        self.pi_is_converged()

        self.policy = self.new_policy

    def pi_plan_offline(self):
        """
        Plan using Policy Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!

        self.pi_initialise()
        while True:
            self.pi_iteration()

            # NOTE: pi_iteration is always called before pi_is_converged
            if self.pi_is_converged():
                break

    def pi_select_action(self, state: GameState):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        # TODO: Implement code to return the optimal action for the given state (based on your stored PI policy) here.
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.

        return self.policy[state]

    # === Helper Methods ===============================================================================================
    def possible_player_positions(self):
        """
        Return a list of possible player positions
        :return: list of possible player positions
        """
        return [(r, c) for r in range(self.game_env.n_rows) for c in range(self.game_env.n_cols) 
                if (self.game_env.grid_data[r][c] not in self.game_env.COLLISION_TILES)]

    def possible_gem_states(self):
        """
        Return a list of possible gem states
        :return: list of possible gem states
        """
        return tuple(gems for gems in itertools.product([0, 1], repeat=self.game_env.n_gems))
    
    def is_absorbing(self, state: GameState):
        # Check if the given state is absorbing
        return state.row == -1 or state.col == -1

