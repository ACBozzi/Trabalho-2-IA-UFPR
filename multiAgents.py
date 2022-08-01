# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
	"""
	A reflex agent chooses an action at each choice point by examining
	its alternatives via a state evaluation function.

	The code below is provided as a guide.  You are welcome to change
	it in any way you see fit, so long as you don't touch our method
	headers.
	"""


	def getAction(self, gameState: GameState):
		"""
		You do not need to change this method, but you're welcome to.

		getAction chooses among the best options according to the evaluation function.

		Just like in the previous project, getAction takes a GameState and returns
		some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
		"""
		# Collect legal moves and successor states
		legalMoves = gameState.getLegalActions()

		# Choose one of the best actions
		scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
		bestScore = max(scores)
		bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
		chosenIndex = random.choice(bestIndices) # Pick randomly among the best

		"Add more of your code here if you want to"

		return legalMoves[chosenIndex]

	def evaluationFunction(self, currentGameState: GameState, action):
		"""
		Design a better evaluation function here.

		The evaluation function takes in the current and proposed successor
		GameStates (pacman.py) and returns a number, where higher numbers are better.

		The code below extracts some useful information from the state, like the
		remaining food (newFood) and Pacman position after moving (newPos).
		newScaredTimes holds the number of moves that each ghost will remain
		scared because of Pacman having eaten a power pellet.

		Print out these variables to see what you're getting, then combine them
		to create a masterful evaluation function.
		"""
		# Useful information you can extract from a GameState (pacman.py)
		successorGameState = currentGameState.generatePacmanSuccessor(action)
		newPos = successorGameState.getPacmanPosition()
		newFood = successorGameState.getFood()
		newGhostStates = successorGameState.getGhostStates()
		newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

		"*** YOUR CODE HERE ***"
		#minimizar a distância até a comida
		if len(newFood.asList()) == currentGameState.getFood().count():
			score = 99999
			for f in newFood.asList():
				if manhattanDistance(f , newPos) < score :
					score = manhattanDistance(f, newPos)
		else:
			score = 0
			# o impacto a medidada que os fantasmas se aproximam
		for ghost in newGhostStates:
			score += 5 ** (2 - manhattanDistance(ghost.getPosition(), newPos))
		return -score


def scoreEvaluationFunction(currentGameState: GameState):
	"""
	This default evaluation function just returns the score of the state.
	The score is the same one displayed in the Pacman GUI.

	This evaluation function is meant for use with adversarial search agents
	(not reflex agents).
	"""
	return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
	"""
	This class provides some common elements to all of your
	multi-agent searchers.  Any methods defined here will be available
	to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

	You *do not* need to make any changes here, but you can if you want to
	add functionality to all your adversarial search agents.  Please do not
	remove anything, however.

	Note: this is an abstract class: one that should not be instantiated.  It's
	only partially specified, and designed to be extended.  Agent (game.py)
	is another abstract class.
	"""

	def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
		self.index = 0 # Pacman is always agent index 0
		self.evaluationFunction = util.lookup(evalFn, globals())
		self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
 	
	def minimax(self, state, depth, agent = 0, maxing = True):
		if depth == 0 or state.isWin() or state.isLose():
			return self.evaluationFunction(state), Directions.STOP
		actions = state.getLegalActions(agent)

		if maxing == True:
			scores = [self.minimax(state.generateSuccessor(agent, action), depth - 1, 1, False)[0] for action in actions]
			bestScore = max(scores)
			bestIndices = [i for i in range(len(scores)) if scores[i] == bestScore]
			return bestScore, actions[bestIndices[0]]

		elif maxing == False:
			scores = []

			if agent == state.getNumAgents() - 1:
				scores = [self.minimax(state.generateSuccessor(agent, action), depth - 1, 0, True)[0] for action in actions]

			else:
				scores = [self.minimax(state.generateSuccessor(agent, action), depth, agent + 1, False)[0] for action in actions]

			bestScore = min(scores)
			bestIndices = [i for i in range(len(scores)) if scores[i] == bestScore]
			return bestScore, actions[bestIndices[0]]
			
			
	def getAction(self, gameState: GameState):
		return self.minimax(gameState, self.depth * 2, 0, True)[1]
		util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
	Your minimax agent with alpha-beta pruning (question 3)
	"""

	def getAction(self, gameState):
		"""
		  Returns the minimax action using self.depth and self.evaluationFunction
		"""
		"*** YOUR CODE HERE ***"

		"""
			AB: receives a state an agent(0,1,2...) and current depth
			AB: return a list:[cost,action]
			Example with depth: 3
			That means pacman played 3 times and all ghosts 3 times
			AB: Cuts some nodes. That means we can use higher depth in the same
			time of minimax algorithm in lower depth
		"""

		def AB(gameState,agent,depth,a,b):
			result = []

			# Terminate state #
			if not gameState.getLegalActions(agent):
				return self.evaluationFunction(gameState),0

			# Reached max depth #
			if depth == self.depth:
				return self.evaluationFunction(gameState),0

			# All ghosts have finised one round: increase depth #
			if agent == gameState.getNumAgents() - 1:
				depth += 1

			# Calculate nextAgent #

			# Last ghost: nextAgent = pacman #
			if agent == gameState.getNumAgents() - 1:
				nextAgent = self.index

			# Availiable ghosts. Pick next ghost #
			else:
				nextAgent = agent + 1

			# For every successor find minmax value #
			for action in gameState.getLegalActions(agent):
				if not result: # First move
					nextValue = AB(gameState.generateSuccessor(agent,action),nextAgent,depth,a,b)

					# Fix result #
					result.append(nextValue[0])
					result.append(action)

					# Fix initial a,b (for the first node) #
					if agent == self.index:
						a = max(result[0],a)
					else:
						b = min(result[0],b)
				else:
					# Check if minMax value is better than the previous one #
					# Chech if we can overpass some nodes                   #

					# There is no need to search next nodes                 #
					# AB Prunning is true                                   #
					if result[0] > b and agent == self.index:
						return result

					if result[0] < a and agent != self.index:
						return result

					previousValue = result[0] # Keep previous value
					nextValue = AB(gameState.generateSuccessor(agent,action),nextAgent,depth,a,b)

					# Max agent: Pacman #
					if agent == self.index:
						if nextValue[0] > previousValue:
							result[0] = nextValue[0]
							result[1] = action
							# a may change #
							a = max(result[0],a)

					# Min agent: Ghost #
					else:
						if nextValue[0] < previousValue:
							result[0] = nextValue[0]
							result[1] = action
							# b may change #
							b = min(result[0],b)
			return result

		# Call AB with initial depth = 0 and -inf and inf(a,b) values      #
		# Get an action                                                    #
		# Pacman plays first -> self.index                                 #
		return AB(gameState,self.index,0,-float("inf"),float("inf"))[1]


#FAZER
class ExpectimaxAgent(MultiAgentSearchAgent):
	"""
	  Your expectimax agent (question 4)
	"""

	def getAction(self, gameState):
		"""
		Returns the expectimax action using self.depth and self.evaluationFunction
		All ghosts should be modeled as choosing uniformly at random from their
		legal moves.
		"""
		"*** YOUR CODE HERE ***"
		return self.getActionExpectimax(gameState, self.depth, 0)[1]

	
	def getActionExpectimax(self, gameState, depth, agentIndex):

		agentNum = gameState.getNumAgents()
		if depth == 0 or gameState.isWin() or gameState.isLose():
			eResult = self.evaluationFunction(gameState)
			return (eResult, '')

		else:
			maxAct = ''
			nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()

			if agentIndex == agentNum - 1:
				depth -= 1

			if agentIndex == 0:
				maxAlp = float('-inf')

			else:
				maxAlp = 0

			maxAct = ''
			nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()

			for action in gameState.getLegalActions(agentIndex):
				gState = gameState.generateSuccessor(agentIndex, action)
				result = self.getActionExpectimax(gState, depth, nextAgentIndex)

				if agentIndex == 0:
					if result[0] > maxAlp:
						maxAlp = result[0]
						maxAct = action
				else:
					maxAlp += 1.0/len(gameState.getLegalActions(agentIndex)) * result[0]
					maxAct = action

		return (maxAlp, maxAct)

def betterEvaluationFunction(currentGameState: GameState):
	"""
	Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
	evaluation function (question 5).

	DESCRIPTION: <write something here so we know what you did>
	"""
	"*** YOUR CODE HERE ***"
	 # Setup information to be used as arguments in evaluation function
	pacman_position = currentGameState.getPacmanPosition()
	ghost_positions = currentGameState.getGhostPositions()

	food_list = currentGameState.getFood().asList()
	food_count = len(food_list)
	capsule_count = len(currentGameState.getCapsules())
	closest_food = 1

	game_score = currentGameState.getScore()

	# Find distances from pacman to all food
	food_distances = [manhattanDistance(pacman_position, food_position) for food_position in food_list]

	# Set value for closest food if there is still food left
	if food_count > 0:
		closest_food = min(food_distances)

	# Find distances from pacman to ghost(s)
	for ghost_position in ghost_positions:
		ghost_distance = manhattanDistance(pacman_position, ghost_position)

		# If ghost is too close to pacman, prioritize escaping instead of eating the closest food
		# by resetting the value for closest distance to food
		if ghost_distance < 2:
			closest_food = 99999

	features = [1.0 / closest_food,
				game_score,
				food_count,
				capsule_count]

	weights = [10,
			   200,
			   -100,
			   -10]

	# Linear combination of features
	return sum([feature * weight for feature, weight in zip(features, weights)])

# Abbreviation
better = betterEvaluationFunction
