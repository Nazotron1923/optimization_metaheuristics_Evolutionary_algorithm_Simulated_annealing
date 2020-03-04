def save(sol, given_function, filename="run_{it}.csv", n_run = 0, fmt = "{val};{sol}\n"):

    f_v = given_function(sol)
    with open(filename.format(it=n_run), 'a') as fd:
        fd.write( fmt.format(val=f_v, sol=sol) )

    return f_v
