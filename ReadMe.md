# Dynamics
This package is an ensemble of code in order to simulate dynamical time evolution using Netket. 

## Frequency
Class making things easier to instantiate a frequency schedule for the Hamiltonian. It generates a callable directly compatible with the Hamiltonian. Each schedule can be manipulated with number (addition, multiplication, time shifting).

It is also possible to append a schedule after the other, as long as the two schedules inherit from the `Frequency` module. The result is again a `Frequency` object with all the previously stated properties.

Some default schedules are implemented, such as constant, linear, quadratic and cubic schedules. There is also the possibility to define a schedule given an array of `t` and `f` values to be extrapolated afterwards. 

## Time-dependent Hamiltonian
Class generalizing the creation of a time-dependent operator made up of multiple sub-operators. For an Hamiltonian of the form $$\hat{\mathcal{H}}(t) = \sum_i f_i(t) \hat{h}_i$$, where $f_i$ are frequency schedules previously defined and $\hat{h}_i$ the operator they tune, the object can be instantiated as `H = TimeDependentHamiltonian([h], [f])`. 

It is then possible to call the Hamiltonian simply by `H(t)`. It also possesses decorators as `to_sparse`, `to_dense`, `to_jax_operator` which modifies the list of sub-operators. 



## Callbacks
Many often used callbacks to keep in the same place. They can be of general use (`callback_parameters`, `CallbackSampler`). 

### Dynamics specific
Directly linked to the `TimeDependentHamiltonian`, we have the `callback_frequencies` which reports all frequencies during the evolution. 
For any Hamiltonian, the TDVP error $R^2$ can be efficiently (since all values are stores) estimated and stored using `callback_R2`. 

For some evolutions, having a non-constant time-step `dt` can be of use. The `DynamicalTimeStep` takes care of this by modifying and reporting the value of the time step all along the evolution, given a callable time-step schedule. The default is set to a constant value of `1e-2`. Some predefined schedules are possible to use, like `constant_dt`, `linear_dt` and `well_dt`. 

## Models
Some often-used models, i.e. implementations of the Jastrow.
In particular, it is possible to define a $N$-body factorized Jastrow $$W_{ijkl ..} = V_{ij} V_{jk} V_{kl} \dots$$ in one line, with `JasMultipleBodies(features=(1,2,4))` (here for a $1$, $2$ and $4$ body interaction). To use a mean-field instead of a $1$-body Jastrow, replace with `JMFMultipleBodies(features=(1,2,4))` simply (the rest remains the same). 

Be aware that this factorized form of the Jastrow yields zero gradients in zero.