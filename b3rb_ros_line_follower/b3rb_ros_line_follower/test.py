import torch

model = torch.load("/home/abdul/cognipilot/cranium/src/b3rb_ros_line_follower/b3rb_ros_line_follower/best.pt", map_location="cpu")

print(type(model))

if isinstance(model, dict):
    print(model.keys())