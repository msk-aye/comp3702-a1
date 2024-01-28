from game_env import GameEnv
import heapq
import functools

"""
solution.py

Where the UCS and A* search algorithms are implemented.

COMP3702 Assignment 1 "Dragon Game" Support Code
"""

class Node:

    def __init__(self, state, parent, previousAction, pathCost, gameEnv):
        self.state = state
        self.gameEnv = gameEnv
        self.parent = parent
        self.previousAction = previousAction
        self.pathCost = pathCost

    def __eq__(self, other):
        return self.state == other.state and self.pathCost == other.pathCost

    def __hash__(self):
        return hash(self.state)

    def __lt__(self, other):
        return self.pathCost < other.pathCost

    def get_pos(self):
        return self.state.get_pos()

    def get_actions(self):
        actions = []
        node = self

        while (node.parent):
            actions.append(node.previousAction)
            node = node.parent

        return actions[::-1]

    def get_successors(self):
        children = []
        for action in self.gameEnv.ACTIONS:
            valid, newState = self.gameEnv.perform_action(self.state, action)

            if valid:
                children.append([newState, action])

        return children


class Solver:

    def __init__(self, game_env: GameEnv):
        self.gameEnv = game_env
        self.initState = self.gameEnv.get_init_state()
        self.initNode = Node(self.initState, None, None, 0, self.gameEnv)

    # === Uniform Cost Search ==========================================================================================
    def search_ucs(self, verbose = True):
        """
        Find a path which solves the environment using Uniform Cost Search (UCS).
        """
        container = [(0, self.initNode)]
        heapq.heapify(container)
        visited = {self.gameEnv.get_init_state(): 0}
        n_expanded = 0

        while len(container) > 0:
            n_expanded += 1
            _, node = heapq.heappop(container)

            # check if this state is the goal
            if self.gameEnv.is_solved(node.state):
                if verbose:
                    print(f'Visited Nodes: {len(visited)},\t\tExpanded Nodes: {n_expanded},\t\t'
                        f'Nodes in Container: {len(container)}')
                    print(f'Cost of Path (with Costly Moves): {node.pathCost}')
                return node.get_actions()

            # add unvisited (or visited at higher path cost) successors to container
            successors = node.get_successors()
            for state, action in successors:
                pathCost = node.pathCost + self.gameEnv.ACTION_COST[action]
                if state not in visited.keys() or pathCost < visited[state]:
                    newNode = Node(state, node, action, pathCost, self.gameEnv)
                    visited[state] = newNode.pathCost
                    heapq.heappush(container, (newNode.pathCost, newNode))

        return None

    # === A* Search ====================================================================================================
    def preprocess_heuristic(self):
        """
        Perform pre-processing (e.g. pre-computing repeatedly used values) necessary for your heuristic,
        """
        # Implement code for any preprocessing required by your heuristic here (if your heuristic)

    @functools.cache
    def estimate_cost(self, start, end):
        """
        Gives a rough estimate of the cost of the player moving from (start) to (end)
        """
        startRow, startCol = start
        endRow, endCol = end

        rowDiff = endRow - startRow
        colDiff = abs(endCol - startCol)

        # Same row, meaning cost is equal to walking
        if rowDiff == 0:
            cost = colDiff

        # End row is higher, meaning cost is equal to jumping + walking
        elif rowDiff < 0:
            rowDiff = abs(rowDiff)
            cost = colDiff + (rowDiff * 2)

        # End row is lower, meaning cost is equal to gliding + dropping or walking
        elif rowDiff > 0:
            colDiff = colDiff // 3
            colRem =  colDiff % 3

            if colDiff <= rowDiff:
                if colRem == 1:
                    cost = (colDiff * 1.2) + (colRem * 0.7) + ((rowDiff - colDiff) * 0.5 / 3)
                else:
                    cost = (colDiff * 1.2) + colRem + ((rowDiff - colDiff - (colRem/2)) * 0.5 / 3)

            else:
                rowDiff = rowDiff // 3
                rowRem = rowDiff % 3
                if rowRem == 1:
                    cost = (rowDiff * 0.2) + (rowRem * -0.3) + colDiff
                else:
                    cost = (rowDiff * 0.2) + colDiff

        return cost

    @functools.cache
    def compute_heuristic(self, state):
        """
        Compute a heuristic value h(n) for the given state.
        """
        playerPos = state.get_pos()
        targetPos = self.gameEnv.exit_row, self.gameEnv.exit_col
        costToPlayer = 0
        costToTarget = 0
        gemCount = state.n_uncollected_gems()

        for gemPos in self.gameEnv.gem_positions:
            if state.gem_status[self.gameEnv.gem_positions.index((gemPos[0], gemPos[1]))] == 0:
                costToPlayer += self.estimate_cost(playerPos, gemPos)
                costToTarget += self.estimate_cost(gemPos, targetPos)

        if gemCount != 0:
            return (costToPlayer + costToTarget) / gemCount

        else:
            return self.estimate_cost(playerPos, targetPos)

    def search_a_star(self, verbose = True):
        """
        Find a path which solves the environment using A* Search.
        """
        # Implement your A* search code here.

        container = [(0, self.initNode)]
        heapq.heapify(container)
        visited = {self.gameEnv.get_init_state(): 0}
        n_expanded = 0

        while len(container) > 0:
            n_expanded += 1
            _, node = heapq.heappop(container)

            # check if this state is the goal
            if self.gameEnv.is_solved(node.state):
                if verbose:
                    print(f'Visited Nodes: {len(visited)},\t\tExpanded Nodes: {n_expanded},\t\t'
                        f'Nodes in Container: {len(container)}')
                    print(f'Cost of Path (with Costly Moves): {node.pathCost}')
                return node.get_actions()

            # add unvisited (or visited at higher path cost) successors to container
            successors = node.get_successors()
            for state, action in successors:
                pathCost = node.pathCost + self.gameEnv.ACTION_COST[action]
                if state not in visited.keys() or pathCost < visited[state]:
                    newNode = Node(state, node, action, pathCost, self.gameEnv)
                    visited[state] = newNode.pathCost
                    heapq.heappush(container, (newNode.pathCost + self.compute_heuristic(state), newNode))

        return None