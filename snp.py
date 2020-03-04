#encoding: utf-8
import math
import numpy as np
import matplotlib.pyplot as plt
import mymodul

from sho import make, algo, iters, plot, num, bit, pb

########################################################################
# Interface
########################################################################

if __name__=="__main__":
    import argparse

    # Dimension of the search space.
    d = 2

    can = argparse.ArgumentParser()

    can.add_argument("-n", "--nb-sensors", metavar="NB", default=3, type=int,
            help="Number of sensors")

    can.add_argument("-r", "--sensor-range", metavar="RATIO", default=0.3, type=float,
            help="Sensors' range (as a fraction of domain width, max is √2)")

    can.add_argument("-w", "--domain-width", metavar="NB", default=30, type=int,
            help="Domain width (a number of cells)")

    can.add_argument("-i", "--iters", metavar="NB", default=100, type=int,
            help="Maximum number of iterations")

    can.add_argument("-gs", "--generation-size", metavar="NB", default=40, type=int,
            help="The generation size")

    can.add_argument("-s", "--seed", metavar="VAL", default=None, type=int,
            help="Random pseudo-generator seed (none for current epoch)")

    solvers = ["num_greedy","bit_greedy", "recuitSimule", "genetic"]
    can.add_argument("-m", "--solver", metavar="NAME", choices=solvers, default="num_greedy",
            help="Solver to use, among: "+", ".join(solvers))

    can.add_argument("-t", "--target", metavar="VAL", default=30*30, type=float,
            help="Objective function value target")

    can.add_argument("-y", "--steady-delta", metavar="NB", default=50, type=float,
            help="Stop if no improvement after NB iterations")

    can.add_argument("-e", "--steady-epsilon", metavar="DVAL", default=0, type=float,
            help="Stop if the improvement of the objective function value is lesser than DVAL")

    can.add_argument("-a", "--variation-scale", metavar="RATIO", default=0.1, type=float,
            help="Scale of the variation operators (as a ration of the domain width)")

    can.add_argument("--tinit", default=5000, type=float,
            help="parameter T init for the algorithm recuitSimule")

    can.add_argument("-al", "--alpha", default=0.5, type=float,
            help="parameter alpha init for the algorithm recuitSimule")

    can.add_argument("--n_run", metavar="N run", default=0, type=int,
            help="Random pseudo-generator seed (none for current epoch)")

    the = can.parse_args()

    # Minimum checks.
    assert(0 < the.nb_sensors)
    assert(0 < the.sensor_range <= math.sqrt(2))
    assert(0 < the.domain_width)
    assert(0 < the.iters)

    # Do not forget the seed option,
    # in case you would start "runs" in parallel.
    np.random.seed(the.n_run)

    # Weird numpy way to ensure single line print of array.
    np.set_printoptions(linewidth = np.inf)


    # Common termination and checkpointing.

    Nnb_it = int((the.iters - 100) / (2*the.generation_size)) if the.solver == 'genetic' else the.iters
    print('Solver: {}, nb_it: {}.\n'.format(the.solver , Nnb_it))

    num_objective_function = num.cover_sum
    bit_objective_function = bit.cover_sum

    history = []
    iters = make.iter(
                iters.several,
                agains = [
                    make.iter(iters.max,
                        nb_it = Nnb_it),
                    # make.iter(iters.save,
                    #     filename = the.solver+".csv",
                    #     fmt = "{it} ; {val} ; {sol}\n"),
                    make.iter(iters.log,
                        fmt="\r{it} {val}"),
                    make.iter(iters.history,
                        history = history),
                    make.iter(iters.target,
                        target = the.target),
                    iters.steady(the.steady_delta, the.steady_epsilon)
                ]
            )

    # Erase the previous file.
    with open(the.solver+".csv", 'w') as fd:
        fd.write("# {} {}\n".format(the.solver,the.domain_width))

    val,sol,sensors = None,None,None
    if the.solver == "num_greedy":
        val,sol = algo.greedy(
                make.func(
                            mymodul.save,
                            given_function = make.func(
                                                        num_objective_function,
                                                        domain_width = the.domain_width,
                                                        sensor_range = the.sensor_range,
                                                        dim = d * the.nb_sensors),
                            filename = the.solver+"{it}.csv",
                            n_run = the.n_run),
                make.init(num.rand,
                    dim = d * the.nb_sensors,
                    scale = the.domain_width),
                make.neig(num.neighb_square,
                    scale = the.variation_scale*3,
                    domain_width = the.domain_width),
                iters
            )
        sensors = num.to_sensors(sol)

    elif the.solver == "bit_greedy":
        val,sol = algo.greedy(
                make.func(bit_objective_function,
                    domain_width = the.domain_width,
                    sensor_range = the.sensor_range,
                    dim = d * the.nb_sensors),
                make.init(bit.rand,
                    domain_width = the.domain_width,
                    nb_sensors = the.nb_sensors),
                make.neig(bit.neighb_square,
                    scale = the.variation_scale*2,
                    domain_width = the.domain_width),
                iters
            )
        sensors = bit.to_sensors(sol)

    elif the.solver == "recuitSimule":
        val,sol = algo.recuitSimule(
                make.func(
                        mymodul.save,
                        given_function = make.func(num_objective_function,
                            domain_width = the.domain_width,
                            sensor_range = the.sensor_range,
                            dim = d * the.nb_sensors),
                        filename = the.solver+"{it}.csv",
                        n_run = the.n_run
                        ),

                make.init(num.rand,
                    dim = d * the.nb_sensors,
                    scale = the.domain_width),
                make.neig(num.neighb_square,
                    scale = the.variation_scale,
                    domain_width = the.domain_width),
                iters,
                the.tinit,
                the.alpha
            )
        sensors = num.to_sensors(sol)

    elif the.solver == "genetic":
        F_func = make.func(
                    mymodul.save,
                    given_function = make.func(
                                                num_objective_function,
                                                domain_width = the.domain_width,
                                                sensor_range = the.sensor_range,
                                                dim = d * the.nb_sensors),
                    filename = the.solver+"{it}.csv",
                    n_run = the.n_run)

        val,sol = algo.genetic(
                F_func,

                make.init(num.generation_g,
                    dim = d*the.nb_sensors,
                    scale = the.domain_width,
                    N = the.generation_size,
                    sensor_range = the.sensor_range*the.domain_width,
                    func = F_func,
                    neighb = make.neig(
                                num.neighb_square,
                                scale = the.variation_scale,
                                domain_width = the.domain_width)),
                make.neig(
                            num.neighb_square,
                            scale = the.variation_scale,
                            domain_width = the.domain_width/2),
                iters,
                make.tournament(num.g_tournament,
                    t = 4),
                make.reproduction(num.g_reproduction,
                    crossover_rate = 0.8,
                    n_point = 2),
                make.mutation(num.g_mutation,
                mutation_rate = 0.1,
                max_x = the.domain_width)
            )
        sensors = num.to_sensors(sol)
    # Fancy output.
    print("\n{} : {}".format(val,sensors))

    shape=(the.domain_width, the.domain_width)

    fig = plt.figure()

    if the.nb_sensors ==1 and the.domain_width <= 50:
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)

        f = make.func(num.cover_sum,
                        domain_width = the.domain_width,
                        sensor_range = the.sensor_range * the.domain_width)
        plot.surface(ax1, shape, f)
        plot.path(ax1, shape, history)
    else:
        ax2=fig.add_subplot(111)

    domain = np.zeros(shape)
    domain = pb.coverage(domain, sensors,
            the.sensor_range * the.domain_width)
    domain = plot.highlight_sensors(domain, sensors)
    ax2.imshow(domain)

    # plt.show()
