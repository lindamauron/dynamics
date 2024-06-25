# Dynamics
This package is an ensemble of code in order to simulate dynamical time evolution using Netket. 

## Frequency
Class making things easier to instantiate a frequency schedule for the Hamiltonian. It generates a callable directly comatible with the Hamiltonian. Each schedule can be manipulated with number (addition, multiplication, time shifting).

It is also possible to append a schedule after the other, as long as the two schedules inherit from the `Frequency` module. The result is again a `Frequency` object with all the previously stated properties.

Some default schedules are implemented, such as constant, linear, quadratic and cubic schedules. There is also the possibility to define a schedule given an array of `t` and `f` values to be extrapolated afterwards. 