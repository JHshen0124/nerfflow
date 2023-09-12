import commentjson as json
import tinycudann as tcnn
import torch
import numpy as np

#for split nerf
def j20():
    pass

def j30():
    pass

#for fully-fused
def j20j30(input_data, out_dim, encoding_config, network_config, batch, net_weight, encoding_weight,density_first=True):
    # kernel definition
    encoding = tcnn.Encoding(3, encoding_config)
    network = tcnn.Network(encoding.n_output_dims, out_dim, network_config)
    network.state_dict['param']=net_weight
    encoding.state_dict['param']=encoding_weight
    model = torch.nn.Sequential(encoding, network)
    for k in range(batch):
        # kernel execution
        feature_and_density = model(input_data)
        if density_first:
            density = feature_and_density[:3]
            feature = feature_and_density[3:]
        else:
            density = feature_and_density[-3:]
            feature = feature_and_density[:-3]
    return density, feature

#for split nerf
def j21():
    pass

def j31():
    pass

#for fully-fused
def j21j31(feature, feature_dim, encoding_config, batch, net_weight, encoding_weight, network_config):
    # kernel execution 
    encoding = tcnn.Encoding(3, encoding_config)
    network = tcnn.Network(encoding.n_output_dims+feature_dim, 3, network_config)# rgb output
    model = torch.nn.Sequential(encoding, network)
    encoding.state_dict["param"]=encoding_weight
    network.state_dict["param"]=net_weight
    # kernel execution
    for k in range(batch):
        color = model(torch.concatenate([encoding,feature]))
    return color

def j00(transform_matrices,f):
    return transform_matrices[f][:3,-1]

def j01(transform_matrices,f,H,W,K):
    # 生成坐标
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H)) 
    i = i.t()
    j = j.t()
    c2w=transform_matrices[f]
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    return rays_d# H*W*3
    
def j1(o,direction,loop_d,delta_distance):
    #请注意需要duplicate j01的数据到整个光线
    return o+loop_d*delta_distance*direction

#只针对一条光线
def j4(framebuffer,color_list,density_list,distance_list,h,w,f):
    delta_distance = distance_list[1:]-distance_list[:-1]
    # 求透明度
    alpha_is = 1.0 - torch.exp(-density_list * delta_distance)
    # 求光线穿透率T
    T_is = torch.cumprod(1.0 - alpha_is + 1e-10, -1)
    T_is = torch.roll(T_is, 1, -1)
    T_is[..., 0] = 1.0 # Ti=1，说明此处光线一定能通过，通过公式反推desity为0
    # 求权重weight
    w_is = T_is * alpha_is
    C_rs = (w_is[..., None] * color_list).sum(dim=-2)
    framebuffer[f][h][w]=C_rs

# for single-gpu
def nerf_fully_fused_loop(F,H,W,D,delta_distance, K,transform_matrics):
    batch = 0
    for f in range(F):
        j00()
        j01()
        for h in range(H):
            for w in range(W):
                j20()
                for d in range(D):
                    batch += j1()
                    if batch == B:
                        j21()
                        j30()
                        batch -= j31()
                j4()



if __name__ == "__main__":
    
    with open("config.json") as f:
        config = json.load(f)
    # test benchmark
    test_focal,test_H,test_W,test_F= 7, 1024, 1024, 2
    test_K = np.array([
            [test_focal, 0, 0.5*test_W],
            [0, test_focal, 0.5*test_H],
            [0, 0, 1]
        ])
    test_transforms = np.array([[[0.886, 0, -0.466, 0.5],
                        [0, 1, 0,0.3],
                        [0.466, 0, 0.886,1]],
                        [[0.967, 0, 0.255,2],
                        [0, 0.983, -0.181,0],
                        [-0.255, 0.181, 0.967,1]]])
    # test j01
    print(j01(test_transforms,0,test_H,test_W,test_K).shape)
