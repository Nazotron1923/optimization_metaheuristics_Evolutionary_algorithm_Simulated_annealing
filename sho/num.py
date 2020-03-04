import math
import numpy as np
import random
from . import pb

########################################################################
# Objective functions
########################################################################

# Decoupled from objective functions, so as to be used in display.
def to_sensors(sol):
    """Convert a vector of n*2 dimension to an array of n 2-tuples.

    >>> to_sensors([0,1,2,3])
    [(0, 1), (2, 3)]
    """
    assert(len(sol)>0)
    sensors = []
    for i in range(0,len(sol),2):
        sensors.append( ( int(math.floor(sol[i])), int(math.floor(sol[i+1])) ) )
    return sensors


def cover_sum(sol, domain_width, sensor_range, dim):
    """Compute the coverage quality of the given vector."""
    assert(0 < sensor_range <= domain_width * math.sqrt(2))
    assert(0 < domain_width)
    assert(dim > 0)
    # print("len(sol)", len(sol))
    # print("dim", dim)
    assert(len(sol) >= dim)

    domain = np.zeros((domain_width,domain_width))
    sensors = to_sensors(sol)
    cov = pb.coverage(domain, sensors, sensor_range*domain_width)
    s = np.sum(cov)
    assert(s >= len(sensors))
    return s


########################################################################
# Initialization
########################################################################

def rand(dim, scale):
    """Draw a random vector in [0,scale]**dim."""
    return np.random.random(dim) * scale


########################################################################
# Neighborhood
########################################################################

def neighb_square(sol, scale, domain_width):
    """Draw a random vector in a square of witdh `scale`
    around the given one."""
    assert(0 < scale <= 1)
    new = sol + (np.random.random(len(sol)) * scale*domain_width - scale*domain_width/2)
    return new



########################################################################
# Genetic
########################################################################
# generate new generation of N
def generation(dim, scale, N):
    """generate new generation of N members."""
    return (np.random.random((N, dim)) * scale)


# generate new generation of N
def generation_g(dim, scale, N, sensor_range, func, neighb):
    """generate new generation of N members."""

    first_sol = np.random.normal(scale/2, scale/3, (dim, ))
    tmp_uniform_l = np.random.uniform(sensor_range, 2*sensor_range, (dim,))
    tmp_uniform_r = np.random.uniform(scale - 2*sensor_range, scale - sensor_range, (dim,))

    first_sol[(first_sol < 2*sensor_range)] = tmp_uniform_l[first_sol < 2*sensor_range]
    first_sol[first_sol > scale-2*sensor_range] = tmp_uniform_r[first_sol > scale-2*sensor_range]
    first_sol = first_sol.tolist()

    T = 5000
    al = 0.5
    best_sol = first_sol
    best_val = func(best_sol)
    val, sol = best_val, best_sol
    i = 0
    while i < 100:
        r = neighb(best_sol)
        r_val = func(r)
        dE =  val - r_val
        p = np.random.uniform(0, 1)
        if (dE < 0) or p < np.exp(-dE/T):
            sol = r
            val = r_val

        if val > best_val:
            j = 0
            best_sol = sol
            best_val = val

        T = al*T
        i+=1


    gereration = []
    for i in range(0, N):
        gereration.append(neighb(best_sol).tolist())

    return gereration
    # return  (np.random.random((N, dim)) * scale).tolist()

def g_tournament(population, fit , t):
    """to write."""
    N = len(population)
    after_tournament = [population[np.argmax(fit)]]
    ind = range(N)
    rng = int(N/2) if (N%2 == 0) else (int(N/2) + 1)
    # for i in range(rng):
    while len(after_tournament) < rng:
        ind_t = random.sample(ind, t)
        candidat_val = fit[ind_t]
        best = np.argmax(candidat_val)
        tmp = list(population[ind_t[best]])
        if tmp not in after_tournament:
            after_tournament.append(tmp)
    return after_tournament


def g_reproduction(population, fit,  crossover_rate, n_point):
    """to write."""
    reproducted = []

    N = len(population)

    for i in range(int(N*(1/2))):
    # for i in range(N):
        parents = random.sample(population, 3)
        parents[1] = population[np.argmax(fit)]

        # ------------------------------2 point crossover ---------------------
        # points = []
        # while len(points) < n_point:
        #     g = random.randint(1, len(parents[0]) - 2)
        #     if (g not in points) and (random.random() > crossover_rate):
        #             points.append(g)
        # points.sort()
        # child = parents[1][:points[0]] + parents[0][points[0]:points[1]] + parents[1][points[1]:] if random.random() > crossover_rate else parents[0][:points[0]] + parents[1][points[0]:points[1]] + parents[0][points[1]:]
        # ------------------------------2 point crossover ---------------------

        # ------------------------------random crossover ----------------------
        child = []
        for pg in range(len(parents[0])):
            tmp = random.random()
            if tmp < 0.33:
                child.append(parents[0][pg])
            elif tmp > 0.66:
                child.append(parents[1][pg])
            else:
                child.append(parents[2][pg])
        # ------------------------------random crossover ----------------------

        reproducted.append(child)

    return reproducted


def g_mutation(population, best_sol, neighb, mutation_rate, max_x):
    """to write."""
    mutated = np.array(population.copy())

    for i in range(0,len(mutated), 3):
        mutated[i] = neighb(best_sol).tolist()
    x =  len(mutated)
    y =  len(mutated[0])
    mut_prob = np.random.uniform(0, 1, (x,y))
    mut_val = np.random.uniform(0, max_x, (x,y))
    mutated[mut_prob<mutation_rate] = mut_val[mut_prob<mutation_rate]

    return mutated.tolist()
