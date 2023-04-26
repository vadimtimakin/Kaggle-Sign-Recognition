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

# xyz = (xyz.permute(1, 0, 2) - torch.min(xyz, dim=1).values) / (torch.max(xyz, dim=1).values - torch.min(xyz, dim=1).values)
# xyz = xyz.permute(1, 0, 2)

# not_nan_xyz = xyz[~torch.isnan(xyz)]
        # if len(not_nan_xyz) != 0:
        #     not_nan_xyz_mean = not_nan_xyz.mean(0, keepdim=True)
        #     not_nan_xyz_std  = not_nan_xyz.std(0, keepdim=True)
        #     xyz -= not_nan_xyz_mean
        #     xyz /= not_nan_xyz_std

# if random.random() < 0.5:
        #     f = interpolate.interp1d(np.arange(0, xyz.shape[0], 1), xyz, axis=0)
        #     xyz = f(np.clip(np.arange(0, xyz.shape[0], random.uniform(0.67, 1.5)), 0, xyz.shape[0] - 1))
        #     xyz = torch.from_numpy(xyz).float()

# def get_shoulders(self, landmarks, left_bodypart, right_bodypart):
#         left = landmarks[:, left_bodypart, :]
#         right = landmarks[:, right_bodypart, :]
#         center = left * 0.5 + right * 0.5
#         return left, right, center

#     def get_pose_size(self, landmarks, left, right, center, torso_size_multiplier=2.5):
#         torso_size = tf.linalg.norm(right - left)

#         pose_center_new = tf.expand_dims(center, axis=1)
#         pose_center_new = tf.broadcast_to(pose_center_new, [tf.size(landmarks) // (543*2), 543, 2])

#         d = tf.gather(landmarks - pose_center_new, 0, axis=0,
#                         name="dist_to_pose_center")
        
#         max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))

#         pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)

#         return pose_size


#     def normalize_pose_landmarks(self, landmarks):
#         left, right, center = self.get_shoulders(landmarks, 500, 501)
#         pose_center = tf.expand_dims(center, axis=1)

#         pose_center = tf.broadcast_to(pose_center, 
#                                         [tf.size(landmarks) // (17*2), 17, 2])
#         landmarks = landmarks - pose_center

#         pose_size = self.get_pose_size(landmarks, left, center,right)
#         landmarks /= pose_size

#         return landmarks


#     def get_shoulders(self, landmarks, left_bodypart, right_bodypart):
#         left = landmarks[:, left_bodypart, :]
#         right = landmarks[:, right_bodypart, :]
#         center = left * 0.5 + right * 0.5
#         return left, right, center

#     def get_pose_size(self, landmarks, left, right, center, torso_size_multiplier=2.5):
#         torso_size = torch.norm(right - left, dim=1)
#         pose_center_new = center.unsqueeze(1).expand_as(landmarks)
#         d = landmarks - pose_center_new
#         max_dist = torch.norm(d[0], dim=1).max()
#         pose_size = torch.max(torso_size * torso_size_multiplier, max_dist)
#         return pose_size

#     def normalize_pose_landmarks(self, landmarks):
#         left, right, center = self.get_shoulders(landmarks, 500, 501)
#         pose_center = center.unsqueeze(1).expand_as(landmarks)
#         landmarks = landmarks - pose_center
#         pose_size = self.get_pose_size(landmarks, left, right, center)
#         landmarks /= pose_size.unsqueeze(1).unsqueeze(1).expand_as(landmarks)
#         return landmarks
    
#     def load_relevant_data_subset(self, path):
#         data_columns = ["x", "y"]
#         data = pd.read_parquet(path, columns=data_columns)
#         n_frames = int(len(data) / 543)
#         data = data.values.reshape(n_frames, 543, len(data_columns))
#         return data.astype(np.float32)

#     def normalise(self, xyz):
#         xyz = xyz[:, :, :2]

#         L = len(xyz)
#         if L > self.config.model.params.max_length:
#             i = self.offset[L]
#             xyz = xyz[i:i+self.config.model.params.max_length]

#         L = len(xyz)
#         not_nan_xyz = xyz[~torch.isnan(xyz)]
#         if len(not_nan_xyz) != 0:
#             not_nan_xyz_mean = not_nan_xyz.mean(0, keepdim=True)
#             not_nan_xyz_std  = not_nan_xyz.std(0, keepdim=True)
#             xyz -= not_nan_xyz_mean
#             xyz /= not_nan_xyz_std
#         xyz_norm = self.normalize_pose_landmarks(xyz)

#         return xyz, xyz_norm, L


# class ISLDataset(Dataset):
#     """The Isolated Sign Language Dataset."""

#     def __init__(self, df, config, is_train):
#         self.df = df
#         self.is_train = is_train
#         self.config = config

#         offset = (np.arange(1000) - self.config.model.params.max_length) // 2
#         offset = np.clip(offset,0, 1000).tolist()
#         self.offset = nn.Parameter(torch.LongTensor(offset),requires_grad=False)

#         self.LHAND = np.arange(468, 489).tolist()
#         self.RHAND = np.arange(522, 543).tolist() 
#         self.REYE = [
#             33, 7, 163, 144, 145, 153, 154, 155, 133,
#             246, 161, 160, 159, 158, 157, 173,
#         ]
#         self.LEYE = [
#             263, 249, 390, 373, 374, 380, 381, 382, 362,
#             466, 388, 387, 386, 385, 384, 398,
#         ]
#         self.SLIP = [
#             78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
#             191, 80, 81, 82, 13, 312, 311, 310, 415,
#         ]
#         self.SPOSE = (np.array([11,13,15,12,14,16,23,24,])+489).tolist()
#         self.TRIU = [
# 			1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
# 			14, 15, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27, 28,
# 			29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
# 			45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
# 			58, 59, 60, 61, 62, 67, 68, 69, 70, 71, 72, 73, 74,
# 			75, 76, 77, 78, 79, 80, 81, 82, 83, 89, 90, 91, 92,
# 			93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 111,
# 			112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
# 			125, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
# 			145, 146, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165,
# 			166, 167, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
# 			188, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 221,
# 			222, 223, 224, 225, 226, 227, 228, 229, 230, 243, 244, 245, 246,
# 			247, 248, 249, 250, 251, 265, 266, 267, 268, 269, 270, 271, 272,
# 			287, 288, 289, 290, 291, 292, 293, 309, 310, 311, 312, 313, 314,
# 			331, 332, 333, 334, 335, 353, 354, 355, 356, 375, 376, 377, 397,
# 			398, 419,
# 		]

#     def __len__(self):
#         return len(self.df)

#     def get_shoulders(self, landmarks, left_bodypart, right_bodypart):
#         left = landmarks[:, left_bodypart, :]
#         right = landmarks[:, right_bodypart, :]
#         center = left * 0.5 + right * 0.5
#         return left, right, center

#     def get_pose_size(self, landmarks, left, right, center, torso_size_multiplier=2.5):
#         torso_size = torch.norm(right - left, dim=1)
#         pose_center_new = center.unsqueeze(1).expand_as(landmarks)
#         d = landmarks - pose_center_new
#         max_dist = torch.norm(d[0], dim=1).max()
#         pose_size = torch.max(torso_size * torso_size_multiplier, max_dist)
#         return pose_size

#     def normalize_pose_landmarks(self, landmarks):
#         left, right, center = self.get_shoulders(landmarks, 500, 501)
#         pose_center = center.unsqueeze(1).expand_as(landmarks)
#         landmarks = landmarks - pose_center
#         pose_size = self.get_pose_size(landmarks, left, right, center)
#         landmarks /= pose_size.unsqueeze(1).unsqueeze(1).expand_as(landmarks)
#         return landmarks
    
#     def load_relevant_data_subset(self, path):
#         data_columns = ["x", "y"]
#         data = pd.read_parquet(path, columns=data_columns)
#         n_frames = int(len(data) / 543)
#         data = data.values.reshape(n_frames, 543, len(data_columns))
#         return data.astype(np.float32)

#     def normalise(self, xyz):
#         xyz = xyz[:, :, :2]

#         L = len(xyz)
#         if L > self.config.model.params.max_length:
#             i = self.offset[L]
#             xyz = xyz[i:i+self.config.model.params.max_length]

#         return xyz, L
    
#     def do_hflip_hand(self, lhand, rhand):
#         rhand[...,0] *= -1
#         lhand[...,0] *= -1
#         rhand, lhand = lhand, rhand
#         return lhand, rhand

#     def do_hflip_eye(self, leye, reye):
#         reye[...,0] *= -1
#         leye[...,0] *= -1
#         reye, leye = leye, reye
#         return leye, reye

#     def do_hflip_spose(self, spose):
#         spose[...,0] *= -1
#         spose = spose[:,[3,4,5,0,1,2,7,6]]
#         return spose

#     def do_hflip_slip(self, slip):
#         slip[...,0] *= -1
#         slip = slip[:,[10,9,8,7,6,5,4,3,2,1,0]+[19,18,17,16,15,14,13,12,11]]
#         return slip

#     def preprocess(self, xyz, L):
#         L = len(xyz)
#         not_nan_xyz = xyz[~torch.isnan(xyz)]
#         if len(not_nan_xyz) != 0:
#             not_nan_xyz_mean = not_nan_xyz.mean(0, keepdim=True)
#             not_nan_xyz_std  = not_nan_xyz.std(0, keepdim=True)
#             xyz -= not_nan_xyz_mean
#             xyz /= not_nan_xyz_std

#         if self.is_train:
#             xyz = self.augment(
#                 xyz=xyz,
#                 scale=self.config.augmentations.scale,
#                 shift=self.config.augmentations.shift,
#                 degree=self.config.augmentations.degree,
#                 p=self.config.augmentations.p,
#             )

#         lhand = xyz[:,self.LHAND]
#         rhand = xyz[:,self.RHAND]
#         spose = xyz[:,self.SPOSE]
#         leye = xyz[:,self.LEYE]
#         reye = xyz[:,self.REYE]
#         slip = xyz[:,self.SLIP]

#         if self.is_train:
#             if random.random() < 0.5:
#                 lhand, rhand = self.do_hflip_hand(lhand, rhand)
#                 spose = self.do_hflip_spose(spose)
#                 leye, reye = self.do_hflip_eye(leye, reye)
#                 slip = self.do_hflip_slip(slip)

#         xyz[:,self.LHAND] = lhand
#         xyz[:,self.RHAND] = rhand
#         xyz[:,self.SPOSE] = spose
#         xyz[:,self.LEYE] = leye
#         xyz[:,self.REYE] = reye
#         xyz[:,self.SLIP] = slip

#         xyz_norm = self.normalize_pose_landmarks(xyz)

#         lhand_norm = xyz_norm[:,self.LHAND]
#         rhand_norm = xyz_norm[:,self.RHAND]
#         spose_norm = xyz_norm[:,self.SPOSE]
#         leye_norm = xyz_norm[:,self.LEYE]
#         reye_norm = xyz_norm[:,self.REYE]
#         slip_norm = xyz_norm[:,self.SLIP]

#         lhand2 = lhand[:, :21, :2]
#         ld = lhand2.reshape(-1, 21, 1, 2) - lhand2.reshape(-1, 1, 21, 2)
#         ld = np.sqrt((ld ** 2).sum(-1))
#         ld = ld.reshape(L, -1)
#         ld = ld[:,self.TRIU]

#         rhand2 = rhand[:, :21, :2]
#         rd = rhand2.reshape(-1, 21, 1, 2) - rhand2.reshape(-1, 1, 21, 2)
#         rd = np.sqrt((rd ** 2).sum(-1))
#         rd = rd.reshape(L, -1)
#         rd = rd[:,self.TRIU]

#         xyz = torch.cat([
#             lhand,
#             rhand,
#             spose,
#             leye,
#             reye,
#             slip,
#             lhand_norm,
#             rhand_norm,
#             spose_norm,
#             leye_norm,
#             reye_norm,
#             slip_norm,
#         ], 1).contiguous()
#         dxyz = F.pad(xyz[:-1] - xyz[1:], [0, 0, 0, 0, 0, 1])

#         xyz = torch.cat([
#             xyz.reshape(L,-1),
#             dxyz.reshape(L,-1),
#             rd.reshape(L,-1),
#             ld.reshape(L,-1),
#         ], -1)

#         xyz[torch.isnan(xyz)] = 0
#         return xyz
            
#     def augment(
#         self,
#         xyz,
#         scale  = (0.8,1.5),
#         shift  = (-0.1,0.1),
#         degree = (-15,15),
#         p=0.5
#     ):
        
#         if random.random() < p:
#             if scale is not None:
#                 scale = np.random.uniform(*scale)
#                 xyz = scale*xyz
    
#         if random.random() < p:
#             if shift is not None:
#                 shift = np.random.uniform(*shift)
#                 xyz = xyz + shift

#         if random.random() < p:
#             if degree is not None:
#                 degree = np.random.uniform(*degree)
#                 radian = degree / 180 * np.pi
#                 c = np.cos(radian)
#                 s = np.sin(radian)
#                 rotate = np.array([
#                     [c,-s],
#                     [s, c],
#                 ]).T
#                 xyz[...,:2] = xyz[...,:2] @rotate
            
#         return xyz

#     def __getitem__(self, idx):
#         sample = self.df.iloc[idx]
#         pq_file = f'{self.config.paths.path_to_folder}{sample.path}'
#         xyz = self.load_relevant_data_subset(pq_file)

#         xyz = torch.from_numpy(xyz).float()
#         xyz, L = self.normalise(xyz)
#         xyz = self.preprocess(xyz, L)

#         return {
#             "features": xyz,
#             "labels": sample.label,
#         }

# class BasedPartyNet(nn.Module):

#     def __init__(self, max_length, embed_dim, num_point, num_head, num_class, num_block):
#         super().__init__()

#         pos_embed = positional_encoding(max_length, embed_dim)
#         self.max_length = max_length

#         self.pos_embed = nn.Parameter(pos_embed)
#         self.embed_dim = embed_dim

#         self.cls_embed = nn.Parameter(torch.zeros((1, embed_dim)))

#         self.x_embed = nn.Sequential(
#             nn.Linear(num_point, embed_dim * 2),
#             nn.LayerNorm(embed_dim * 2),
#             nn.Hardswish(),
#             nn.Dropout(0.4),
#             nn.Linear(embed_dim * 2, embed_dim),
#             nn.LayerNorm(embed_dim),
#             nn.Hardswish(),
#         )

#         self.encoder = nn.ModuleList([
#             TransformerBlock(
#                 embed_dim + num_point,
#                 num_head,
#                 embed_dim + num_point,
#             ) for _ in range(num_block)
#         ])
#         self.logit = ArcMarginProduct_subcenter(self.embed_dim + num_point, num_class)

#     def forward(self, inputs):
#         xyz, x_mask = pack_seq(inputs, self.max_length)

#         B,L,_ = xyz.shape
#         x = self.x_embed(xyz)
#         x = x + self.pos_embed[:L].unsqueeze(0)

#         x = torch.cat([
#             self.cls_embed.unsqueeze(0).repeat(B,1,1),
#             x
#         ],1)

#         x_mask = torch.cat([
#             torch.zeros(B,1).to(x_mask),
#             x_mask
#         ],1)

#         xyz = torch.cat([
#             torch.zeros(B, 1, 828).to(x_mask),
#             xyz
#         ],1)

#         x = torch.cat([x, xyz], dim=2)

#         for block in self.encoder:
#             x = block(x,x_mask)

#         x = F.dropout(x,p=0.4,training=self.training)
#         x_mask = x_mask.unsqueeze(-1)
#         x_mask = 1-x_mask.float()
#         x = (x*x_mask).sum(1)/x_mask.sum(1)
#         logit = self.logit(x)

#         return logit


# class SpatialAttention(nn.Module):
#     def __init__(self):
#         super(SpatialAttention, self).__init__()

#         self.conv1 = nn.Linear(2, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=2, keepdim=True)
#         max_out, _ = torch.max(x, dim=2, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=2)
#         x = self.conv1(x)
#         return self.sigmoid(x)


# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, in_planes))
#         self.max_pool = nn.AdaptiveMaxPool2d((1, in_planes))
           
#         self.fc = nn.Sequential(nn.Linear(in_planes, in_planes // 16, 1),
#                                nn.Hardswish(),
#                                nn.Linear(in_planes // 16, in_planes))
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out)


# class TCM(nn.Module):
#     def __init__(self, n_segment, fold_dim):
#         super().__init__()
#         self.n_segment = n_segment
#         self.fold_div = fold_dim

#     def forward(self, x):
#         nt, c, l = x.size()
#         n_batch = nt // self.n_segment
#         x = x.view(n_batch, self.n_segment, c, l)
        
#         fold = c // self.fold_div

#         out = torch.zeros_like(x)
#         out[:, :-1, :fold] = x[:, 1:, :fold]
#         out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
#         out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        
#         out = out.view(nt, c, l)
#         return out


#     def normalize_hand(self, keypoints):
#         x_max, _ = torch.max(keypoints[:, :, 0], dim=1)
#         x_min, _ = torch.min(keypoints[:, :, 0], dim=1)
#         y_max, _ = torch.max(keypoints[:, :, 1], dim=1)
#         y_min, _ = torch.min(keypoints[:, :, 1], dim=1)
        
#         width = x_max - x_min
#         height = y_max - y_min
#         center_x = (x_max + x_min) / 2
#         center_y = (y_max + y_min) / 2
        
#         box_width = torch.max(0.1 * width, 0.1 * height)
#         box_height = torch.max(0.1 * width, 0.1 * height)
        
#         bbox_left = center_x - box_width / 2
#         bbox_top = center_y - box_height / 2
        
#         normalized_keypoints = (keypoints - torch.stack([bbox_left, bbox_top], dim=1).unsqueeze(1)) / torch.stack([box_width, box_height], dim=1).unsqueeze(1)
        
#         normalized_keypoints -= 0.5

#         return normalized_keypoints