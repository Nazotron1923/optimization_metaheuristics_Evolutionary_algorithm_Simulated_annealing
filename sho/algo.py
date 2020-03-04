########################################################################
# Algorithms
########################################################################
import numpy as np

def random(func, init, again):
    """Iterative random search template."""
    best_sol = init()
    best_val = func(best_sol)
    val,sol = best_val,best_sol
    i = 0
    while again(i, best_val, best_sol):
        sol = init()
        val = func(sol)
        if val >= best_val:
            best_val = val
            best_sol = sol
        i += 1
    return best_val, best_sol


def greedy(func, init, neighb, again):
    """Iterative randomized greedy heuristic template."""
    best_sol = init()
    best_val = func(best_sol)
    val,sol = best_val,best_sol
    i = 1
    while again(i, best_val, best_sol):
        sol = neighb(best_sol)
        val = func(sol)
        # Use >= and not >, so as to avoid random walk on plateus.
        if val >= best_val:
            best_val = val
            best_sol = sol
        i += 1
    return best_val, best_sol

# TODO add a simulated-annealing template.
def recuitSimule(func, init, neighb, again, Tinit, al):
    T = Tinit
    best_sol = init()
    best_val = func(best_sol)
    val, sol = best_val, best_sol
    i = 0
    while again(i, best_val, best_sol):

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
    return best_val, best_sol


# TODO add a population-based stochastic heuristic template.

def genetic(func, init, neighb, again, tournament, reproduction, mutation):

    # generate new population
    population = init()
    # fit all population
    fit = np.apply_along_axis(func, 1, population)

    best_val = np.max(fit)
    best_sol = population[np.argmax(fit)]
    val, sol = best_val, best_sol

    i = 0 # n-th generation


    while again(i, best_val, best_sol):
        N = len(population)
        # do tournament
        after_tournament = tournament(population, fit)
        # do reproduction
        reproducted = reproduction(population, fit)
        # do mutation

        mutated = mutation(reproducted, best_sol, neighb)

        # ---------------------- Super-mutation--------------------------------
        super_mut_child = []
        super_mut_child_fit = []

        best_sol_rc = best_sol
        best_val_rc = best_val
        val_rc, sol_rc = best_val_rc, best_sol_rc
        i_rc = 0 # iteration parameter for recuitSimule
        T = 5000 # temperature parameter for recuitSimule
        al = 0.5 # alpha parameter for recuitSimule
        while i_rc < N:
            r = neighb(best_sol_rc)
            r_val = func(r)
            dE =  val_rc - r_val
            p = np.random.uniform(0, 1)
            if (dE < 0) or p < np.exp(-dE/T):
                sol_rc = r
                val_rc = r_val

            if val_rc > best_val_rc:
                best_sol_rc = sol_rc
                best_val_rc = val_rc
                best_val, best_sol = best_val_rc, best_sol_rc

            T = al*T
            i_rc+=1

        super_mut_child.append(list(best_sol_rc))
        super_mut_child_fit.append(best_val_rc)
        fit_ch = np.array(super_mut_child_fit)
        # ---------------------- Super-mutation--------------------------------


        # do new generation
        i+=1 # next generation


        population = after_tournament + mutated[:len(mutated) - 1]
        fit = np.apply_along_axis(func, 1, population)
        population = population + super_mut_child
        fit = np.concatenate((fit, fit_ch))


        tmp = np.max(fit)
        if tmp > best_val:
            best_val = tmp
            best_sol = population[np.argmax(fit)]



    return func(best_sol), best_sol
