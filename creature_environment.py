import random
import numpy as np
from PIL import Image
import cv2
from math import pow
import neat
import pickle
import threading

rand = lambda x: random.randint(0, x)

SIZE = 50
N_EPISODES = 200

PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

d = {1: (255, 175, 0),  # blueish color
     2: (0, 255, 0),  # green
     3: (0, 0, 255)}  # red


class Environment(object):
    '''
    Set up environment for the creatures.
    '''
    def __init__(self, genomes, config, num_food=250):
        self.output_env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        self.env = np.zeros((SIZE, SIZE), dtype=np.uint8)
        self.creature_list = [Creature(genome, config) for genome in genomes]
        self.creature_dict = dict([((creature.x, creature.y), creature) for creature in self.creature_list])
        self.food_list = [Food() for _ in range(num_food)]
        self.food_dict = dict([((food.x, food.y), food) for food in self.food_list])
        # self.food_coor = map(str, self.food_dict.keys())
        self.points = 0
        self.results = []

    def return_array(self):

        self.output_env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        self.env = np.zeros((SIZE, SIZE), dtype=np.uint8)

        for food in self.food_list:
            if not food.is_eaten:
                self.output_env[food.x][food.y] = d[food.color]
                self.env[food.x][food.y] = food.color

        for creature in self.creature_list:
            self.output_env[creature.x][creature.y] = d[creature.color]
            self.env[creature.x][creature.y] = creature.color

        return self.output_env

    def play_all(self, display=False, show_when=-1): 
        

        for episode in range(N_EPISODES):
            if show_when > -1:
                if not episode % show_when:
                    display = True
                else:
                    display = False
            
            for creature in self.creature_list:
                self.play_a_move(creature)


            if display:
                img = Image.fromarray(self.return_array(), 'RGB')
                img = img.resize((900, 900))
                cv2.imshow("image", np.array(img))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                _ = self.return_array()
        
        if display:
            cv2.destroyAllWindows()
    
        self.results = [(creature.points, creature.genome) for creature in self.creature_list]


    def solo_play(self):
        creature_moves = [threading.Thread(target=self.play_a_move, args=(creature,), daemon=True) for creature in self.creature_list]
        counter = 1
        for i in creature_moves:
            print(f"Thread {counter}")
            counter += 1
            i.start()

    def play_a_move(self, creature):
        creature.action(self.food_dict)
        # output_env = self.return_array()
        if self.env[creature.x][creature.y] == FOOD_N:
            food = creature.eat(self.food_dict[(creature.x, creature.y)])
            self.food_dict[(creature.x, creature.y)] = food
            self.food_list = list(self.food_dict.values())

    def __getitem__(self, x):
        return self.output_env[x]

    def __setitem__(self, x, y):
        self.output_env[x] = y

class Food(object): 
    def __init__(self):
        self.x = rand(SIZE-1)
        self.y = rand(SIZE-1)
        self.is_eaten = False
        self.color = FOOD_N

    def __str__(self):
        return str(self.__dict__)

class Creature(object):
    def __init__(self, genome, config, **kwargs):
        self.id = random.random()
        self.x = rand(SIZE-1)
        self.y = rand(SIZE-1)
        self.points = 0
        self.hunger = 0
        self.gives_consent = False
        self.enemy = False #random.choices([True, False], weights=[0.1, 0.9])[0]
        self.color = PLAYER_N if not self.enemy else ENEMY_N
        self.genome = genome
        self.config = config
        self.neural_network = neat.nn.FeedForwardNetwork.create(genome, config)

        self.__dict__ = {**self.__dict__, **kwargs}

    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        penalty = 0.5
        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
            self.points -= penalty
        elif self.x > SIZE-1:
            self.x = SIZE-1
            self.points -= penalty
        if self.y < 0:
            self.y = 0
            self.points -= penalty
        elif self.y > SIZE-1:
            self.y = SIZE-1 
            self.points -= penalty

    def action(self, dict_of_coor, choice=None):
        '''
        Gives us 4 total movement options. (0,1,2,3)
        And toggle for consent.
        '''
        if choice == None:
            """ l = [-1, 0, 1]
            coor = [(self.x+i, self.y+j) for i in l for j in l]
            coor.remove((self.x, self.y))
            input = [dict_of_coor[c[0]][c[1]] for c in coor]
            for i in range(len(input)):
                if input[i] == 1:
                    input[i] = 0
 """
            input = self.compute_euclidian_dist(dict_of_coor)
            
            choice = np.argmax(self.neural_network.activate(input))      

        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=0, y=1)
        elif choice == 6:
            self.move(x=-1, y=0)
        elif choice == 7:
            self.move(x=0, y=-1)
        """ elif choice == 8:
            self.gives_consent = not self.gives_consent """

    def compute_euclidian_dist(self, food_dict):
        keys = list(food_dict.keys())
        return keys[np.argmax([pow(pow(self.x-food_x, 2) + pow(self.y-food_y, 2), 0.5)  for food_x, food_y in keys])]

    def eat(self, food):
        if not food.is_eaten:
            food.is_eaten = True
            self.points += 5
        return food
    
    def __add__(self, other):
        other_type = type(other)

        if other_type == Creature and other.gives_consent:
            crossover_id = pow((self.id * other.id) + 2*self.id*other.id, 0.5)
            return Creature(self.genome, self.config,id=crossover_id, x=self.x+1, y=self.y+1)

    def __str__(self):
        return str(self.__dict__)


def evolutionary_driver(n=50):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config')

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))

    # Run until we achieve n.
    winner = p.run(eval_genomes, n=n)

    # dump
    pickle.dump(winner, open('winner.pkl', 'wb'))

def eval_genomes(genomes, config):
    _, genomes = zip(*genomes)
    env = Environment(genomes, config)
    env.play_all(display=False)#, show_when=10)
    results = env.results 

    top_score = 0
    for points, genome in results:
        fitness = points
        genome.fitness = -1 if fitness == 0 else fitness

        if top_score < points:
            top_score = points

    #print score
    print('The top score was:', top_score)
        
        


    


'''
def play(env, display=True):

    for _ in range(N_EPISODES):

        env.solo_play()

        if display:
            img = Image.fromarray(env.return_array(), 'RGB')
            img = img.resize((900, 900))
            cv2.imshow("image", np.array(img))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if display:
        cv2.destroyAllWindows()
'''
""" 
env = Environment()

env.play_all() """

evolutionary_driver(2000)