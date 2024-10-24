# Dynamics
This package is an ensemble of code in order to simulate dynamical time evolution using Netket. 

## Frequency
Class making things easier to instantiate a frequency schedule for the Hamiltonian. It generates a callable directly compatible with the Hamiltonian. Each schedule can be manipulated with number (addition, multiplication, time shifting).

It is also possible to append a schedule after the other, as long as the two schedules inherit from the `Frequency` module. The result is again a `Frequency` object with all the previously stated properties.

Some default schedules are implemented, such as constant, linear, quadratic and cubic schedules. There is also the possibility to define a schedule given an array of `t` and `f` values to be extrapolated afterwards. 

## Time-dependent Hamiltonian
Class generalizing the creation of a time-dependent operator made up of multiple sub-operators. For an Hamiltonian of the form $$\hat{\mathcal{H}}(t) = \sum_i f_i(t) \hat{h}_i$$, where $f_i$ are frequency schedules previously defined and $\hat{h}_i$ the operator they tune, the object can be instantiated as `H = TimeDependentHamiltonian([h], [f])`. 

It is then possible to call the Hamiltonian simply by `H(t)`. It also possesses decorators as `to_sparse`, `to_dense`, `to_jax_operator` which modifies the list of sub-operators. 

## Drivers
There are two new drivers compatible with netket's utilities.

### ExactEvolution 
A simple exact evolution driver that uses sparse matrices for the hamiltonian to apply it on a dense state. 

### SemiExactTDVP
It decomposes the Hamiltonian in a diagonal and off-diagonal part $H(t) = H_x(t) + H_z(t)$ and applies the diagonal part exactly to the parameters $$U_z(t) \exp( \sum_{ij} W_{ij} z_i z_j + \sum_i V_i z_i ) = \exp( \sum_{ij} \tilde{W}_{ij} z_i z_j + \sum_i \tilde{V}_i z_i ).$$ Since this will change for all hamiltonians and models, the functions `Uz` and `Hx` must be `dispatch`ed for each setting.

In particular, there are multiple ways to adress the application of $U_z$ (see this [paper](https://arxiv.org/abs/2410.05955) for example), e.g.
1. $U_z(t_1,t_2) = \exp( -i H_z(t_1) (t_2-t_1)) $
2. $U_z(t_1,t_2) = \exp( -i \int_{t_1}^{t_2} d\tau H_z(\tau) ) $
3. $U_z(t_1,t_2) = \exp( -i H_z(\frac{t_2+t_1}{2}) (t_2-t_1)) $

which should be equivalent for $dt \to 0$. 

A typical dispatch will look like
```python
@nkt.driver.Hx.dispatch
def my_hx(
        hamiltonian : MyHamiltonian, t, **kwargs
):
    return hamiltonian.frequencies[0](t) * hamiltonian.operators[0]

@nkt.driver.Uz.dispatch
def my_uz(state, driver, machine : my_model, t1, t2):
    dw = jax.tree.map(lambda x: 0*x, state.parameters)
    coeff = -1j * driver.generator.frequencies[1](t1) * (t2-t1)
    dw['W2']['kernel'] = coeff * driver.generator.J
    dw['W1']['kernel'] = coeff * driver.generator.h
    return optax.apply_updates(state.parameters, dw)
```


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