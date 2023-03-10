#importing packages
import numpy as np 
import copy 

#Now, define the class PuzzleNode:
class PuzzleNode:
    """
    Class PuzzleNode: Provides a structure for performing A* search for the n^2-1 puzzle
    """
    # Class constructor
    def __init__(self,state,coord,fval,gval,parent=None):
        self.state = state
        self.coord = coord
        self.fval = fval
        self.gval = gval
        self.parent = parent
        self.pruned = False

    # Comparison function based on f cost
    def __lt__(self,other):
        return self.fval < other.fval

    # Convert to string
    def __str__(self):
        return str(self.state)
    
# Misplaced tiles heuristic
def h1(state):
    """
    This function returns the number of misplaced tiles, given the board state
    Input:
        -state: the board state as a list of lists
    Output:
        -h: the number of misplaced tiles
    """
    n = len(state)
    #generates goal state of the board
    goal = [[i+j for i in range(n)] for j in range(0, n**2, n)]

    #counts the number of incorrectly placed tiles
    num_misplaced = 0
    for i in range(n):
        for j in range(n):
            if state[i][j] != goal[i][j] and state[i][j] != 0:
                num_misplaced += 1
    
    return num_misplaced

# Manhattan distance heuristic
def h2(state):
    """
    This function returns the Manhattan distance from the solved state, given the board state
    Input:
        -state: the board state as a list of lists
    Output:
        -h: the Manhattan distance from the solved configuration
    """
    #simplifies the list to a flat list
    state_simplified = single_list(state)
    n = len(state_simplified)
    
    #creates goal state
    goal_state = []
    for i in range(n):
        goal_state.append(i)

    #computes the manhattan distance for each digit 
    return sum(abs(b%3 - g%3) + abs(b//3 - g//3)
        for b, g in ((state_simplified.index(i), goal_state.index(i)) for i in range(1, n)))

# If you implement more than 3 heuristics, then add any extra heuristic functions onto the end of this list.
heuristics = [h1, h2]

##extra helper functions

#function to check if the state is nxn
def check_puzzle(state):
    """
    This function checks the puzzle for correct format
    Input:
        -state: the board state as a list of lists
    Output:
        -error = -1 
        - optimal path = none
        - steps counter, nodes expanded, max frontier = 0
    """
    #the size of all nested lists should be equal to the size of the list
    n = len(state)
    for i in range(n):
        if len(state[i]) != n:
            return -1
        
def repeated_number(state):    
    """
    This function checks the puzzle for correct format
    Input:
        -state: the board state as a list of lists
    Output:
        -error = -1 
        - optimal path = none
        - steps counter, nodes expanded, max frontier = 0
    """
    #the size of all nested lists should be equal to the size of the list
    n = len(state)
    list = single_list(state)
    for i in range(len(list)):
        if list.count(list[i]) > 1:
            return -1
        if list[i] > n**2 -1:
            return -1


#function to convert list of lists to single list
def single_list(state):
    """
    This function returns a flat list, given a list with nested lists
    Input:
        -state: the board state as a list of lists
    Output:
        -state_list = a flat list with he current configuration
    """
    state_list = []
    for sublist in state:
        for item in sublist:
            state_list.append(item)
    return state_list

def find_index(lst, value):
    """
    This function returns the coordinates of empty space, given state and 0
    Input:
        -state: the board state as a list of lists
        -value: value of empty space (0)
    Output:
        -(i,j): tuple of int coordinates of empty space
    """
    for i, sublist in enumerate(lst):
        if value in sublist:
            j = sublist.index(value)
            return (i, j)
        
def swap_elements(state, next_coord,cur_coord):
    """
    This function returns an updated state space, given current state, next coordinates of
    empty space and current coordinates of empty space
    Input:
        -state: the board state as a list of lists
        -next coordinates: future coordinates of empty space
        -current coordinates: the coordinates of empty space
    Output:
        -state: updated state space
    """
    #makes a copy of the current state
    next_state = copy.deepcopy(state)

    temp = next_state[cur_coord[0]][cur_coord[1]]
    #swap the digit at next coordinate with zero
    next_state[cur_coord[0]][cur_coord[1]] = next_state[next_coord[0]][next_coord[1]]
    next_state[next_coord[0]][next_coord[1]] = temp
    return next_state

def moves(state, start):
    """
    This function returns a list of possible moves that empty space can make
    Input:
        -state: the board state as a list of lists
        -start: current coordinates of empty space
    Output:
        -moves: list  of possible moves
    """
    #conditioning on empty space at borders
    moves_orth = []
    if start[0] > 0:
        moves_orth.append([-1,0])
    if start[0] < len(state) - 1:
        moves_orth.append([1,0])
    if start[1] > 0:
        moves_orth.append([0,-1])
    if start[1] < len(state) - 1:
        moves_orth.append([0,1])
    
    return moves_orth
            
def new_coord(coordinates,m):
    """
    This function returns new coordinates for 
    Input:
        -coordinates: current coordinates of empty space
        -m: the possible move
    Output:
        -new coordinates: updating coordinates of emoty space after the move
    """
    new_coord = (coordinates[0]+m[0],coordinates[1]+m[1])
    return new_coord

#3Adapted from CS152, Session 3.1
##R. Shekhar

# Main solvePuzzle function.
def solvePuzzle(state, heuristic):
    """This function should solve the n**2-1 puzzle for any n > 2 (although it may take too long for n > 4)).
    Inputs:
        -state: The initial state of the puzzle as a list of lists
        -heuristic: a handle to a heuristic function.  Will be one of those defined in Question 2.
    Outputs:
        -steps: The number of steps to optimally solve the puzzle (excluding the initial state)
        -exp: The number of nodes expanded to reach the solution
        -max_frontier: The maximum size of the frontier over the whole search
        -opt_path: The optimal path as a list of list of lists.  That is, opt_path[:,:,i] should give a list of lists
                    that represents the state of the board at the ith step of the solution.
        -err: An error code.  If state is not of the appropriate size and dimension, return -1.  For the extention task,
          if the state is not solvable, then return -2
    """
    #print('Starting the function')
    
    #creating a goal board
    format_check = check_puzzle(state)
    input_check = repeated_number(state)
    if format_check or input_check == -1:
        return 0, 0, 0,None, -1
    
    
    n = len(state)
    goal = [[i+j for i in range(n)] for j in range(0, n**2, n)]
    #print('Goal state loaded')

    #finding the current coordinates of empty space
    start = find_index(state,0)
    #print('0 is currently at', start)
    #creating the first node
    start_node = PuzzleNode(state,start,heuristic(state),0)
    #print('The starting state is', start_node)

# Dictionary with current cost to reach all visited nodes
    costs_db = {str(start_node.state):start_node}

# Frontier, stored as a Priority Queue to maintain ordering
    from queue import PriorityQueue
    
    
    frontier = PriorityQueue()
    frontier.put(start_node)
    length_frontier = 1
    max_frontier = length_frontier


# next moves

# Begin A* Tree Search
    step_counter = 0
    nodes_expanded = 0

    while not frontier.empty():
        # Take the next available node from the priority queue
        cur_node = frontier.get()
        #print('current state', cur_node.state)

        if cur_node.pruned:
            continue # Skip if this node has been marked for removal

        # Check if we are at the goal
        if cur_node.state == goal:   
            break

        # Expand the node in the orthogonal and diagonal directions
        #print(cur_node.coord)
        valid_moves = moves(state,cur_node.coord)
        step_counter += 1
        gval = cur_node.gval + 1 #tentative cost value for the child node
        #print('valid moves', valid_moves)

        #for every valid moce
        for m in valid_moves:
            #print("moving empty space to",m)
            #compute the new coordinate of empty space
            next_coord = new_coord(cur_node.coord,m)
            #move the empty space to the new coordinate
            next_state = swap_elements(cur_node.state,next_coord,cur_node.coord)
            #print("the resulting state after moving", next_state)
            hval = heuristic(next_state)
            #print("the new state",next_state)
            #generate respective child node with the new state
            child_node = PuzzleNode(next_state,next_coord,gval+hval,gval,cur_node)


# If the child node is already in the cost database (i.e. explored) then see if we need to update the path.  In a graph search, we wouldn't even bother exploring it again.
            if str(child_node.state) in costs_db:
                if costs_db[str(child_node.state)].gval > gval:
                    costs_db[str(child_node.state)].pruned = True # Mark existing value for deletion from frontier
                else:
                    #print("disregarding the child")
                    continue  # ignore this child, since a better path has already been found previously
                

            next_node = child_node
            nodes_expanded += 1
            #PuzzleNode(child,child_coordinate[child],gval+hval+1,gval+1,cur_node) # Create new node for child

            frontier.put(next_node)
            length_frontier += 1
            costs_db[str(next_state)] = next_node #Mark the node as explored

            if length_frontier>max_frontier:
                max_frontier = length_frontier


# Reconstruct the optimal path
    optimal_path = [cur_node.state]
    while cur_node.parent:
        optimal_path.append((cur_node.parent).state)
        cur_node = cur_node.parent
    optimal_path = optimal_path[::-1]
    #print(f"A* search completed in {step_counter} steps\n")
    #print(f"A* path length: {len(optimal_path)-1} steps\n")
    #print(f"A* path to goal:\n")
    #print(optimal_path)
    error_code = 0

    return len(optimal_path)-1, nodes_expanded, max_frontier,optimal_path, error_code

## Test for state not correctly defined

incorrect_state = [[0,1,2],[2,3,4],[5,6,7]]
_,_,_,_,err = solvePuzzle(incorrect_state, lambda state: 0)
assert(err == -1)

## Heuristic function tests for misplaced tiles and manhattan distance

# Define the working initial states
working_initial_states_8_puzzle = ([[2,3,7],[1,8,0],[6,5,4]], [[7,0,8],[4,6,1],[5,3,2]], [[5,7,6],[2,4,3],[8,1,0]])

# Test the values returned by the heuristic functions
h_mt_vals = [7,8,7]
h_man_vals = [15,17,18]

for i in range(0,3):
    h_mt = heuristics[0](working_initial_states_8_puzzle[i])
    h_man = heuristics[1](working_initial_states_8_puzzle[i])
    assert(h_mt == h_mt_vals[i])
    assert(h_man == h_man_vals[i])
## A* Tests for 3 x 3 boards
## This test runs A* with both heuristics and ensures that the same optimal number of steps are found
## with each heuristic.

# Optimal path to the solution for the first 3 x 3 state
opt_path_soln = [[[2, 3, 7], [1, 8, 0], [6, 5, 4]], [[2, 3, 7], [1, 8, 4], [6, 5, 0]], 
                 [[2, 3, 7], [1, 8, 4], [6, 0, 5]], [[2, 3, 7], [1, 0, 4], [6, 8, 5]], 
                 [[2, 0, 7], [1, 3, 4], [6, 8, 5]], [[0, 2, 7], [1, 3, 4], [6, 8, 5]], 
                 [[1, 2, 7], [0, 3, 4], [6, 8, 5]], [[1, 2, 7], [3, 0, 4], [6, 8, 5]], 
                 [[1, 2, 7], [3, 4, 0], [6, 8, 5]], [[1, 2, 0], [3, 4, 7], [6, 8, 5]], 
                 [[1, 0, 2], [3, 4, 7], [6, 8, 5]], [[1, 4, 2], [3, 0, 7], [6, 8, 5]], 
                 [[1, 4, 2], [3, 7, 0], [6, 8, 5]], [[1, 4, 2], [3, 7, 5], [6, 8, 0]], 
                 [[1, 4, 2], [3, 7, 5], [6, 0, 8]], [[1, 4, 2], [3, 0, 5], [6, 7, 8]], 
                 [[1, 0, 2], [3, 4, 5], [6, 7, 8]], [[0, 1, 2], [3, 4, 5], [6, 7, 8]]]

astar_steps = [17, 25, 28]
for i in range(0,3):
    steps_mt, expansions_mt, _, opt_path_mt, _ = solvePuzzle(working_initial_states_8_puzzle[i], heuristics[0])
    steps_man, expansions_man, _, opt_path_man, _ = solvePuzzle(working_initial_states_8_puzzle[i], heuristics[1])
    # Test whether the number of optimal steps is correct and the same
    assert(steps_mt == steps_man== astar_steps[i])
    # Test whether or not the manhattan distance dominates the misplaced tiles heuristic in every case
    assert(expansions_man < expansions_mt)
    # For the first state, test that the optimal path is the same
    if i == 0:
        assert(opt_path_mt == opt_path_soln)

## A* Test for 4 x 4 board
## This test runs A* with both heuristics and ensures that the same optimal number of steps are found
## with each heuristic.

working_initial_state_15_puzzle = [[1,2,6,3],[0,9,5,7],[4,13,10,11],[8,12,14,15]]
steps_mt, expansions_mt, _, _, _ = solvePuzzle(working_initial_state_15_puzzle, heuristics[0])
steps_man, expansions_man, _, _, _ = solvePuzzle(working_initial_state_15_puzzle, heuristics[1])
# Test whether the number of optimal steps is correct and the same
assert(steps_mt == steps_man == 9)
# Test whether or not the manhattan distance dominates the misplaced tiles heuristic in every case
assert(expansions_mt >= expansions_man)
