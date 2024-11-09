import torch

model1 = torch.load("/home/zzh/proj/NewWork/results/train/vim_epoch_0.pt")
model2 = torch.load("/home/zzh/proj/NewWork/results/train/722.pt")

dict1 = model1['state_dict']
dict2 = model2['state_dict']


for key in dict1.keys():
    print(key)
print("\n================================\n")


# for key in dict2.keys():
#     value1 = dict1[key].to("cuda:0")
#     value2 = dict2[key].to("cuda:0")

#     if torch.equal(value1, value2):
#         print(f"{key} equal!!!!!!!!!!")
#     else:
#         print(f"{key} not equal")
# print("\n================================\n")


gamma = model1['state_dict']['colorizer.gamma']
print(f"gamma = {gamma}")


