import os
import copy
import sys
import random
from collections import OrderedDict
from skimage import io, filters
from PIL import Image, ImageOps
from pathlib import Path
import math
import numpy as np
import scipy.misc
from timeit import default_timer as timer


class AntMap:
    ''' Keeps track of ant movements for drawing purposes'''
    def __init__(self):
        self.registered_ants = dict()

    def register_ant(self, ant):
        ''' Creates a new turtle-ant and registrs it '''
        new_turtle_ant = Turtle()
        if ant.id % 10 == 0:
            c = 1
        new_turtle_ant.color = c
        self.registered_ants[ant] = new_turtle_ant

    def draw_ant_path(self, ant):
        ''' Draws a real-time map of the ants' movements '''
        ant = self.registered_ants.get(ant)
        if ant is None:
            self.register_ant(ant)
        turtle_ant = registered_ants[ant]
        color('red')


class Ant:
    ''' An ant searches new pixels with the greatest visibility, alerting others of its findings via pheromones'''
    def __init__(self, row, column, colony, idNo, special=False):
        self.row = row
        self.column = column
        self.colony = colony
        self.id = idNo
        self.special = special
        self.memory = OrderedDict()

    def determine_in_bound_moves(self):
        ''' Determine which directions an ant can move in, excluding out-of-bounds '''
        legal_moves = copy.copy(AntImage.directions)

        if self.row == 0:
            del legal_moves['NW']
            del legal_moves['N']
            del legal_moves['NE']

        elif self.row == self.colony.image.num_rows-1:
            del legal_moves['SW']
            del legal_moves['S']
            del legal_moves['SE']

        if self.column == 0:
            del legal_moves['W']
            if 'NW' in legal_moves: del legal_moves['NW']
            if 'SW' in legal_moves: del legal_moves['SW']

        elif self.column == self.colony.image.num_columns-1:
            del legal_moves['E']
            if 'NE' in legal_moves: del legal_moves['NE']
            if 'SE' in legal_moves: del legal_moves['SE']

        return legal_moves

    def calculate_move_probability(self, row, column):
        ''' Determine the probability (numerator only) of making a move towards a given pixel '''

        # Alias variables to resemble formula on paper
        alpha = self.colony.pheromone_weight
        beta = self.colony.visibility_weight
        tau = self.colony.image.pheromones[row, column]
        eta = self.colony.image.visibilities[row, column]

        return (tau ** alpha) * (eta ** beta)

    def evaluate_surroundings_probability(self):
        ''' Calculates an ordered dictionary of probabilities of moving towards adjacent pixels '''
        legal_moves = self.determine_in_bound_moves()
        probabilities = dict()

        for move in legal_moves.values():
            potential_row = self.row + move[0]
            potential_column = self.column + move[1]

            # If a move would put an ant in a position it remembers, then that move is illegal
            if (potential_row, potential_column) in self.memory:
                probabilities[move] = 0
            # If a move would place an ant in a pixel with an exceedingly low visibility, ignore the move
            elif self.colony.image.visibilities[potential_row, potential_column] < self.colony.visibility_threshold:
                probabilities[move] = 0
            else:
                probabilities[move] = self.calculate_move_probability(potential_row, potential_column)
        return probabilities

    @staticmethod
    def normalize_probabilities(probabilities):
        ''' Normalizes a probability distribution by dividing values by their sum
            If division by zero would occur, it signals that the ant should be warped '''

        denominator = sum(probabilities)
        if denominator == 0:
            return None
        else:
            for i in range(len(probabilities)):
                probabilities[i] /= denominator
            return probabilities

    def move(self, probabilities_dictionary):
        ''' Moves the ant in a given direction, leaving a pheromone trail '''

        # First split the probabilities dictionary into two lists, as is required by random.choice()
        moves = list()
        probabilities = list()
        items = probabilities_dictionary.items()
        for item in items:
            moves.append(item[0])
            probabilities.append(item[1])

        # Normalize probabilities, and warp ant if there is no other choice
        probabilities = self.normalize_probabilities(probabilities)
        if probabilities is None:
            self.row = random.randrange(0, self.colony.image.num_rows)
            self.column = random.randrange(0, self.colony.image.num_columns)
            return "WARP"

        # Randomly choose next move
        else:
            choice_index = np.random.choice(len(moves), p=probabilities)
            choice = moves[choice_index]
            self.row += choice[0]
            self.column += choice[1]
            return "WALK"

    def deposit_pheromones(self):
        ''' Calculate the amount of pheromone to deposit in current position '''
        self.colony.image.pheromones_deltas[self.row, self.column] += self.colony.image.visibilities[self.row, self.column]

    def update_memory(self):
        ''' Adds current position to memory, and forgets positions visited long ago '''
        self.memory[(self.row, self.column)] = "VISITED"
        if len(self.memory) > self.colony.memory_length:
            self.memory.popitem(last=False)

    def update(self):
        ''' Calls all necessary functions to update an individual ant's status '''
        probabilities = self.evaluate_surroundings_probability()
        move_type = self.move(probabilities)
        if move_type != "WARP":
            self.deposit_pheromones()
        self.update_memory()


class AntImage:
    ''' Image properties relevant to edge detection '''
    N = (-1, 0)
    NE = (-1, 1)
    E = (0, 1)
    SE = (1, 1)
    S = (1, 0)
    SW = (1, -1)
    W = (0, -1)
    NW = (-1, -1)
    directions = {'N':N, 'NE':NE, 'E':E, 'SE':SE, 'S':S, 'SW':SW, 'W':W, 'NW':NW}

    def __init__(self, values, min_pheromone):
        self.values = values
        self.num_rows = values.shape[0]
        self.num_columns = values.shape[1]
        self.size = self.num_rows * self.num_columns
        self.pheromones = np.full((self.num_rows, self.num_columns), min_pheromone)
        self.pheromones_deltas = np.zeros_like(self.pheromones)
        self.visibilities = np.zeros_like(values, dtype=np.float64)
        self.calculate_visibilities()

    def calculate_visibilities(self):
        ''' Calculates the visibility value for each pixel ants will use as a heuristic '''
        max_value = self.values.max()
        print("debug", self.values.shape)
        for row, column in np.ndindex(self.values.shape):
            # Calculate four potential visibility values: horizontal, vertical and rising/falling diagonals
            v_horizontal = 0
            v_vertical = 0
            v_rising = 0
            v_falling = 0
            is_border_pixel = False

            # Numpy stores pixel information as ui8, so casting to int is necessary to avoid overflow
            # No horizontal visibility if pixel is column edge
            if 0 < column < self.num_columns-1:
                v_horizontal = abs(int(self.values[row, column-1]) - int(self.values[row, column+1]))
            else:
                is_border_pixel = True

            # No vertical visibility if pixel is row edge
            if 0 < row < self.num_rows-1:
                v_vertical = abs(int(self.values[row-1, column]) - int(self.values[row+1, column]))
            else:
                is_border_pixel = True

            # No diagonal visibility if pixel is border pixel
            if not is_border_pixel:
                v_rising = abs(int(self.values[row+1, column-1]) - int(self.values[row-1, column+1]))
                v_falling = abs(int(self.values[row-1, column-1]) - int(self.values[row+1, column+1]))

            # Visibility of a given pixel is the maximum of all linear visibilities, normalized
            self.visibilities[row, column] = float(1/max_value) * max(v_horizontal, v_vertical, v_rising, v_falling)
        #print("DEBUG:\n", self.visibilities)


class Colony:
    ''' The colony describes parameters common to all ants '''
    def __init__(self, image_data, antNo=None, pheromone_weight=2.5, pheromone_evaporation=0.02, pheromone_minimum=0.0001,
                 visibility_weight =2, visibility_threshold=0.08, memory_length=39):
        # Adjustment parameters
        if antNo is None:
            antNo = round(math.sqrt(image_data.shape[0]*image_data.shape[1]))
            print(antNo, "ants")

        self.antNo = antNo
        self.image = AntImage(image_data, pheromone_minimum)
        self.ants = [None] * antNo
        self.pheromone_weight = pheromone_weight # Alpha
        self.pheromone_minimum = pheromone_minimum # Tau_min
        self.pheromone_evaporation = pheromone_evaporation # Rho
        self.visibility_weight = visibility_weight # Beta
        self.visibility_threshold = visibility_threshold # b
        self.memory_length = memory_length


    def initialize_ants(self):
        ''' Randomly creates and places ants on image '''
        for i in range(len(self.ants)):
            row = random.randrange(0, self.image.num_rows)
            column = random.randrange(0, self.image.num_columns)
            self.ants[i] = Ant(row, column, self, i)

    def update_pheromones(self):
        for row, column in np.ndindex(self.image.values.shape):
            # Alias variables to resemble paper
            rho = self.pheromone_evaporation
            tau_old = self.image.pheromones[row, column]
            tau_delta = self.image.pheromones_deltas[row, column]
            tau_min = self.pheromone_minimum

            # Update pheromones
            self.image.pheromones[row, column] = (1-rho)*tau_old + tau_delta
            self.image.pheromones_deltas[row, column] = 0

            # Prevent pheromone from plummeting below set minimum
            if self.image.pheromones[row, column] < tau_min:
                self.image.pheromones[row, column] = tau_min



    def run(self, iterations, steps):
        '''' Executes the ants algorithm to produce an edge image '''
        resultpath = imagepath + "\\final.bmp"
        self.initialize_ants()
        for i in range(iterations):
            print("Iteration:", i+1)
            for ant in self.ants:
                for s in range(steps):
                    ant.update()
            self.update_pheromones()
            generate_image_from_array(path=resultpath, array=self.image.pheromones, id=i)
        print("Completed!")
        #print("Parameters used:" "AntNo = ", len(sel.ants), "Alpha = ", "Beta = ", "Rho = ", "Min Pheromone = ", "Memory Length =")

def generate_image_from_array(path, array, id):
    ''' Saves an image to disc from array with inverted colors '''
    #minimum = array.min()
    #maximum = array.max()
    #translation = interp1d([minimum, maximum], [0, 255])
    #for row, column in np.ndindex(array.shape):
    #    array2[row, column] = np.uint8(translation(array[row, column]))
    print("DEBUG:\n", array)
    array = array.astype(np.uint8)
    #array.shape = (512, 512)
    threshold = filters.threshold_isodata(array, 256)
    for row, column in np.ndindex(array.shape):
        if array[row, column] >= threshold:
            array[row, column] = 0
        else:
            array[row, column] = 255
    #print('debug', array)
    dir_base = os.path.dirname(path)
    #Path(dir_base).mkdir(parents=True, exist_ok=True)
    result = Image.frombytes('L', array.shape, array)
    result.show()
    result.save(str(id) + '.bmp')

def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    ''' Copied from scipy.misc source code '''
    if data.dtype == np.uint8:
        return data

    if high < low:
        raise ValueError("`high` should be larger than `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data * 1.0 - cmin) * scale + 0.4999
    bytedata[bytedata > high] = high
    bytedata[bytedata < 0] = 0
    return np.cast[np.uint8](bytedata) + np.cast[np.uint8](low)

# Filepath globals

filepath = input("Please specify which file you wish to trace edges for: ")
imagepath =  "..\\Images"

if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)

    image = bytescale(io.imread(filepath, as_gray=True))
    print("SHAPE =", image.shape)
    print("original image\n", image)
    result = Image.frombytes('L', image.shape, image)
    colony = Colony(image, antNo=500)
    start = timer()
    colony.run(3, 500)
    end = timer()
    print("Time taken:", end-start)
