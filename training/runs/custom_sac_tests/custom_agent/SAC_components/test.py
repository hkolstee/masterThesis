from actor import Actor
import torch

actor = Actor(0.003, 10, 4, [0.1, 0.1, 0.1, 0.1], [0.6, 0.6, 0.6, 0.6], (256, 256))

print(actor.scale_action(torch.tensor([0.2, 0.2, 0.2, 0.2])))
print(actor.unscale_action(actor.scale_action(torch.tensor([0.2, 0.2, 0.2, 0.2]))))
