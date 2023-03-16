# def pre_process(xyz):
#     lip = xyz[:, LIP]
#     lhand = xyz[:, LHAND]
#     rhand = xyz[:, RHAND]
#     xyz = torch.cat([ #(none, 82, 3)
#         lip,
#         lhand,
#         rhand,
#     ],1)
#     xyz = xyz - xyz[~torch.isnan(xyz)].mean(0,keepdims=True) #noramlisation to common maen
#     xyz = xyz / xyz[~torch.isnan(xyz)].std(0, keepdims=True)
#     xyz[torch.isnan(xyz)] = 0
#     xyz = xyz[:max_length]
#     return xyz