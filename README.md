This is an open-source framework that provides a simple interface for operating with pre-implemented deep reinforcement learning algorithms built on top of PyTorch. The interfaces provided can be used to operate deep RL agents either locally or remotely (via gRPC). Currently, this framework supports the following algorithms

- [x] DQN

## Reference

`cartpole_local_dqn.py` provides a simple example of this framework being used on gymnasium's CartPole-v1 envinronment. `cartpole_grpc_dqn.py` provides an example of how to use the gRPC interface to learn remotely.

The main class that is used in this framework is `TorchDeepRL`, implemented in `servers/`. The gRPC server has an interface very similar to it.

### TorchDeepRL
| Method | Parameters | Explanation |
|---|---|---|
| initialize() | config: dict | Initializes the agent according to the config. Read the config section for information on formatting |
| step() | state: Tensor, reward: Tensor, terminal: bool | Performs an optimization step and returns the selected action for this  |
| experience_and_optimize() | state: Tensor, reward: Tensor, terminal: bool | Performs an optimization step without selecting an action |

### gRPC interface
Note that there is a client implemented in `servers/torch_deep_rl_grpc_client.py` that has the exact same interface as `TorchDeepRL`. This reference is mostly meant for building clients in other programming languages.

| Method | Parameters | Explanation |
|---|---|---|
| initialize() | config: string | Accepts a serialized config dict |
| step() | state: Matrix(dimensions list[int], value: list[float]), reward: float, terminal: bool | State should be given as a flattened matrix, action is returned the same way  |
| experience_and_optimize() | state: Matrix(dimensions list[int], value: list[float]), reward: float, terminal: bool | State should be given as a flattened matrix |

## Config formatting
The config file is a dictionary that specifies the behavior of the agent. The RL implementation is specified by the value of the key "type". It also accepts a variety of other arguments depending on the imeplementation type.

Currently supported implementations are `dqn`.

The following specify the arguments allowed by each implementation type.

### DQN
| Argument | Explanation |
|---|---|
| "action_space": dict | The gym Space that represents the action space of the environment. Read the Space table on `Other specifications` |
| "gamma": float | Value of gamma |
| "tau": float | Value of tau
| "epsilon": dict | The epsilon. Read the Epsilon table on `Other specifications` |
| "batch_size": int | Batch size |
| "grad_clip_value": float | Value to clip gradient. No clipping if not specified |
| "loss": dict | The loss. Read the Loss section on `Other specifications` |
| "optimizer": dict | The optimizer. Read the Optimizer section on `Other specifications` |
| "replay_buffer_size": int | Capacity of the replay buffer |
| "network": list[dict] | list of layer specs for the neural network. Read the Layers section on `Other specifications` |

### Other specifications

These are specifications for dictionaries that are used in the specification of learners. They each have an argument "type" and a corresponding class or function. In the case of classes, all of its initializing parameters can be passed as arguments in this dictionary. When specific arguments are expected, they will be made explicit.

#### Space
| Type | Class |
|---|---|
| discrete | `gym.spaces.Discrete` |

#### Epsilon
You can read `components/epsilon.py` to see how each of these are implemented
| Type | Arguments | Explanation
|---|---|---|
| exp_decrease | "start": float, "end": float, "steps_to_end": int | Decreases the epsilon exponentially over time.

#### Loss
| Type | Function |
|---|---|
| smooth_l1_loss | `torch.nn.SmoothL1Loss` |

#### Optimizer
| Type | Class |
|---|---|
| adamw | `torch.optim.AdamW` |

#### Layers
| Type | Function |
|---|---|
| linear | `torch.nn.Linear` |
| relu | `torch.nn.ReLU` |



