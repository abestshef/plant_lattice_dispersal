# plant_lattice_dispersal
Python code to accompany the manuscript "Using a lattice model with local and global transmission to predict the spread of a plant disease" by Best &amp; Cunniffe

Two separate code files are provided, one for the lattice model (plant_lattice_model.py) and one for the dispersal model (plant_dispersal_model.py). These will both:

1. Run a chosen number of simulations for a chosen error threshold. (Note, the dispersal model runs ~100 times slower than the lattice model, so fewer runs may be possible).
2. Plot the posterior distributions of the model parameters for the chosen threshold.
3. Plot the time-courses against the data for 100 posterior predictive checked (both the determinstic and stochastic verisons for the lattice model).
