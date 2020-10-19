The goal is to train the agents 'Ant-v3' and Humanoid-v3'
from the MuJoCo environments of OpenAI Gym to move on the circumference of a
circle centered on the origin while remaining within a safety region smaller than the radius of the circle.

The Environments were inspired by the circle environments in <https://arxiv.org/pdf/1705.10528.pdf> 
and are directly modified from the implementations
in [OpenAI Gym](https://github.com/openai/gym). 

### Installation
Go to environment directory

```cd mujoco-circle```

then install by

```pip install -e .```
### Loading the Environments
After installation is complete, you can create an instance
of the environment with ```gym.make('mujoco_circle:AntCircle-v0')```
(For HumanoidCircle, replace 'AntCircle-v0' with 
'HumanoidCircle-v0').
### Example
The code snippet below generates an episode of maximum length
1000 using a random policy from the AntCircle environment.
```
import gym

env = gym.make('mujoco_circle:AntCircle-v0')
state = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    if done:
        break
env.close() 
```



