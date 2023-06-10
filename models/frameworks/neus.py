from models.base import ImplicitSurface, NeRF, RadianceNet
from utils import rend_util, train_util

import copy
import functools
import numpy as np
from tqdm import tqdm
from typing import Optional
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

#finetune losses
from criteria.clip_loss import CLIPLoss
from criteria.patchnce_loss import PatchNCELoss
from criteria.contrastive_loss import ContrastiveLoss
from criteria.perp_loss import VGGPerceptualLoss
import random
import gc
# def pdf_phi_s(x: torch.Tensor, s):
#     esx = torch.exp(-s*x)
#     y = s*esx / ((1+esx) ** 2)
#     return y

FIX_MODULE = "radiance_net"
def cdf_Phi_s(x, s):
    # den = 1 + torch.exp(-s*x)
    # y = 1./den
    # return y
    return torch.sigmoid(x*s)


def sdf_to_alpha(sdf: torch.Tensor, s):
    # [(B), N_rays, N_pts]
    cdf = cdf_Phi_s(sdf, s)
    # [(B), N_rays, N_pts-1]
    # TODO: check sanity.
    opacity_alpha = (cdf[..., :-1] - cdf[..., 1:]) / (cdf[..., :-1] + 1e-10)
    opacity_alpha = torch.clamp_min(opacity_alpha, 0)
    return cdf, opacity_alpha


def sdf_to_w(sdf: torch.Tensor, s):
    device = sdf.device
    # [(B), N_rays, N_pts-1]
    cdf, opacity_alpha = sdf_to_alpha(sdf, s)

    # [(B), N_rays, N_pts]
    shifted_transparency = torch.cat(
        [
            torch.ones([*opacity_alpha.shape[:-1], 1], device=device),
            1.0 - opacity_alpha + 1e-10,
        ], dim=-1)
    
    # [(B), N_rays, N_pts-1]
    visibility_weights = opacity_alpha *\
        torch.cumprod(shifted_transparency, dim=-1)[..., :-1]

    return cdf, opacity_alpha, visibility_weights


def alpha_to_w(alpha: torch.Tensor):
    device = alpha.device
    # [(B), N_rays, N_pts]
    shifted_transparency = torch.cat(
        [
            torch.ones([*alpha.shape[:-1], 1], device=device),
            1.0 - alpha + 1e-10,
        ], dim=-1)
    
    # [(B), N_rays, N_pts-1]
    visibility_weights = alpha *\
        torch.cumprod(shifted_transparency, dim=-1)[..., :-1]

    return visibility_weights

class NeuS(nn.Module):
    def __init__(self,
                 variance_init=0.05,
                 speed_factor=1.0,

                 input_ch=3,
                 W_geo_feat=-1,
                 use_outside_nerf=False,
                 obj_bounding_radius=1.0,

                 surface_cfg=dict(),
                 radiance_cfg=dict()):
        super().__init__()
        
        self.ln_s = nn.Parameter(data=torch.Tensor([-np.log(variance_init) / speed_factor]), requires_grad=True)
        self.speed_factor = speed_factor

        #------- surface network
        self.implicit_surface = ImplicitSurface(
            W_geo_feat=W_geo_feat, input_ch=input_ch, obj_bounding_size=obj_bounding_radius, **surface_cfg)
        
        #------- radiance network
        if W_geo_feat < 0:
            W_geo_feat = self.implicit_surface.W
        self.radiance_net = RadianceNet(
            W_geo_feat=W_geo_feat, **radiance_cfg)

        #-------- outside nerf++
        if use_outside_nerf:
            self.nerf_outside = NeRF(input_ch=4, multires=10, multires_view=4, use_view_dirs=True)

    def forward_radiance(self, x: torch.Tensor, view_dirs: torch.Tensor,return_nablas=False):
        _, nablas, geometry_feature = self.implicit_surface.forward_with_nablas(x)
        radiance = self.radiance_net.forward(x, view_dirs, nablas, geometry_feature)
        return radiance

    def forward_s(self):
        return torch.exp(self.ln_s * self.speed_factor)


    def forward(self, x: torch.Tensor, view_dirs: torch.Tensor,return_nablas=False):
        sdf, nablas, geometry_feature = self.implicit_surface.forward_with_nablas(x)
        radiances = self.radiance_net.forward(x, view_dirs, nablas, geometry_feature)
        return radiances, sdf, nablas
    
    def fix_module(self, module_name):
        if module_name is None or module_name == "":
            return
        if module_name == 'implicit_surface':
            for p in self.implicit_surface.parameters():
                p.requires_grad = False
        elif module_name == 'radiance_net':
            for p in self.radiance_net.parameters():
                p.requires_grad = False
        else:
            raise NotImplementedError(f"{module_name} is not a valid module.")
                # print debug info
        print("############DEBUG###############")
        for name, param in self.named_parameters():
            print(name, param.requires_grad)
        print("############DEBUG###############")

def volume_render(
    rays_o, 
    rays_d,
    model: NeuS,
    
    obj_bounding_radius=1.0,
    
    batched = False,
    batched_info = {},

    # render algorithm config
    calc_normal = False,
    use_view_dirs = True,
    rayschunk = 65536,
    netchunk = 1048576,
    white_bkgd = False,
    near_bypass: Optional[float] = None,
    far_bypass: Optional[float] = None,

    # render function config
    detailed_output = True,
    show_progress = False,

    # sampling related
    perturb = False,   # config whether do stratified sampling
    fixed_s_recp = 1/64.,
    N_samples = 64,
    N_importance = 64,
    N_outside = 0,  # whether to use outside nerf
    
    # upsample related
    upsample_algo = 'official_solution',
    N_nograd_samples = 2048,
    N_upsample_iters = 4,
    
    **dummy_kwargs  # just place holder
):
    """
    input: 
        rays_o: [(B,) N_rays, 3]
        rays_d: [(B,) N_rays, 3] NOTE: not normalized. contains info about ratio of len(this ray)/len(principle ray)
    """
    device = rays_o.device
    if batched:
        DIM_BATCHIFY = 1
        B = rays_d.shape[0]  # batch_size
        flat_vec_shape = [B, -1, 3]
    else:
        DIM_BATCHIFY = 0
        flat_vec_shape = [-1, 3]

    rays_o = torch.reshape(rays_o, flat_vec_shape).float()
    rays_d = torch.reshape(rays_d, flat_vec_shape).float()
    # NOTE: already normalized
    rays_d = F.normalize(rays_d, dim=-1)
    
    #! Aug 14:  add return nabla flag for vlosdf compability
    batchify_query = functools.partial(train_util.batchify_query, chunk=netchunk, dim_batchify=DIM_BATCHIFY,
     return_nablas=True)
    
    # ---------------
    # Render a ray chunk
    # ---------------
    def render_rayschunk(rays_o: torch.Tensor, rays_d: torch.Tensor):
        # rays_o: [(B), N_rays, 3]
        # rays_d: [(B), N_rays, 3]
        
        # [(B), N_rays] x 2
        near, far = rend_util.near_far_from_sphere(rays_o, rays_d, r=obj_bounding_radius)
        if near_bypass is not None:
            near = near_bypass * torch.ones_like(near).to(device)
        if far_bypass is not None:
            far = far_bypass * torch.ones_like(far).to(device)
        
        if use_view_dirs:
            view_dirs = rays_d
        else:
            view_dirs = None
    
        prefix_batch = [B] if batched else []
        N_rays = rays_o.shape[-2]

        # ---------------
        # Sample points on the rays
        # ---------------


        # ---------------
        # Coarse Points
        
        # [(B), N_rays, N_samples]
        # d_coarse = torch.linspace(near, far, N_samples).float().to(device)
        # d_coarse = d_coarse.view([*[1]*len(prefix_batch), 1, N_samples]).repeat([*prefix_batch, N_rays, 1])
        _t = torch.linspace(0, 1, N_samples).float().to(device)
        d_coarse = near * (1 - _t) + far * _t
        
        # ---------------
        # Up Sampling
        with torch.no_grad():
            # -------- option 1: directly use
            if upsample_algo == 'direct_use':   # nerf-like
                # [(B), N_rays, N_samples, 3]
                pts_coarse = rays_o.unsqueeze(-2) + d_coarse.unsqueeze(-1) * rays_d.unsqueeze(-2)
                # query network to get sdf
                # [(B), N_rays, N_samples]
                sdf_coarse = model.implicit_surface.forward(pts_coarse)
                # [(B), N_rays, N_samples-1]
                *_, w_coarse = sdf_to_w(sdf_coarse, 1./fixed_s_recp)
                # Fine points
                # [(B), N_rays, N_importance]
                d_fine = rend_util.sample_pdf(d_coarse, w_coarse, N_importance, det=not perturb)
                # Gather points
                d_all = torch.cat([d_coarse, d_fine], dim=-1)
                d_all, d_sort_indices = torch.sort(d_all, dim=-1)

            # -------- option 2: just using more points to calculate visibility weights for upsampling
            # used config: N_nograd_samples
            elif upsample_algo == 'direct_more':
                _t = torch.linspace(0, 1, N_nograd_samples).float().to(device)
                _d = near * (1 - _t) + far * _t
                _pts = rays_o.unsqueeze(-2) + _d.unsqueeze(-1) * rays_d.unsqueeze(-2)
                # _sdf = model.implicit_surface.forward(_pts)
                _sdf = batchify_query(model.implicit_surface.forward, _pts)
                *_, _w = sdf_to_w(_sdf, 1./fixed_s_recp)
                d_fine = rend_util.sample_pdf(_d, _w, N_importance, det=not perturb)
                # Gather points
                d_all = torch.cat([d_coarse, d_fine], dim=-1)
                d_all, d_sort_indices = torch.sort(d_all, dim=-1)


            # -------- option 3: modified from NeuS official implementation: estimate sdf slopes and middle points' sdf
            # https://github.com/Totoro97/NeuS/blob/9dc9275d3a8c7266994a3b9cf9f36071621987dd/models/renderer.py#L131
            # used config: N_upsample_iters
            elif upsample_algo == 'official_solution':
                _d = d_coarse
                _sdf = batchify_query(model.implicit_surface.forward, rays_o.unsqueeze(-2) + _d.unsqueeze(-1) * rays_d.unsqueeze(-2))
                for i in range(N_upsample_iters):
                    prev_sdf, next_sdf = _sdf[..., :-1], _sdf[..., 1:]
                    prev_z_vals, next_z_vals = _d[..., :-1], _d[..., 1:]
                    mid_sdf = (prev_sdf + next_sdf) * 0.5
                    dot_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)
                    prev_dot_val = torch.cat([torch.zeros_like(dot_val[..., :1], device=device), dot_val[..., :-1]], dim=-1)   # jianfei: prev_slope, right shifted
                    dot_val = torch.stack([prev_dot_val, dot_val], dim=-1)  # jianfei: concat prev_slope with slope
                    dot_val, _ = torch.min(dot_val, dim=-1, keepdim=False)  # jianfei: find the minimum of prev_slope and current slope. (forward diff vs. backward diff., or the prev segment's slope vs. this segment's slope)
                    dot_val = dot_val.clamp(-10.0, 0.0)
                    
                    dist = (next_z_vals - prev_z_vals)
                    prev_esti_sdf = mid_sdf - dot_val * dist * 0.5
                    next_esti_sdf = mid_sdf + dot_val * dist * 0.5
                    
                    prev_cdf = cdf_Phi_s(prev_esti_sdf, 64 * (2**i))
                    next_cdf = cdf_Phi_s(next_esti_sdf, 64 * (2**i))
                    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
                    _w = alpha_to_w(alpha)
                    d_fine = rend_util.sample_pdf(_d, _w, N_importance // N_upsample_iters, det=not perturb)
                    _d = torch.cat([_d, d_fine], dim=-1)
                    
                    sdf_fine = batchify_query(model.implicit_surface.forward, rays_o.unsqueeze(-2) + d_fine.unsqueeze(-1) * rays_d.unsqueeze(-2))
                    _sdf = torch.cat([_sdf, sdf_fine], dim=-1)
                    _d, d_sort_indices = torch.sort(_d, dim=-1)
                    _sdf = torch.gather(_sdf, DIM_BATCHIFY+1, d_sort_indices)
                d_all = _d
            else:
                raise NotImplementedError

        # ------------------
        # Calculate Points
        # [(B), N_rays, N_samples+N_importance, 3]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * d_all[..., :, None]
        # [(B), N_rays, N_pts-1, 3]
        # pts_mid = 0.5 * (pts[..., 1:, :] + pts[..., :-1, :])
        d_mid = 0.5 * (d_all[..., 1:] + d_all[..., :-1])
        pts_mid = rays_o[..., None, :] + rays_d[..., None, :] * d_mid[..., :, None]

        # ------------------
        # Inside Scene
        # ------------------
        # sdf, nablas, _ = model.implicit_surface.forward_with_nablas(pts)
        sdf, nablas, _ = batchify_query(model.implicit_surface.forward_with_nablas, pts)
        # [(B), N_ryas, N_pts], [(B), N_ryas, N_pts-1]
        cdf, opacity_alpha = sdf_to_alpha(sdf, model.forward_s())
        # radiances = model.forward_radiance(pts_mid, view_dirs_mid)
        radiances = batchify_query(model.forward_radiance, pts_mid, view_dirs.unsqueeze(-2).expand_as(pts_mid) if use_view_dirs else None)

        # ------------------
        # Outside Scene
        # ------------------
        if N_outside > 0:
            _t = torch.linspace(0, 1, N_outside + 2)[..., 1:-1].float().to(device)
            d_vals_out = far / torch.flip(_t, dims=[-1])
            if perturb:
                _mids = .5 * (d_vals_out[..., 1:] + d_vals_out[..., :-1])
                _upper = torch.cat([_mids, d_vals_out[..., -1:]], -1)
                _lower = torch.cat([d_vals_out[..., :1], _mids], -1)
                _t_rand = torch.rand(_upper.shape).float().to(device)
                d_vals_out = _lower + (_upper - _lower) * _t_rand
            
            d_vals_out = torch.cat([d_mid, d_vals_out], dim=-1) # already sorted
            pts_out = rays_o[..., None, :] + rays_d[..., None, :] * d_vals_out[..., :, None]
            r = pts_out.norm(dim=-1, keepdim=True)
            x_out = torch.cat([pts_out/r, 1./r], dim=-1)
            views_out = view_dirs.unsqueeze(-2).expand_as(x_out[..., :3]) if use_view_dirs else None
            
            #! Aug 14: dummy variable for volsdf return nabla compability
            sigma_out, radiance_out,_ = batchify_query(model.nerf_outside.forward, x_out, views_out)
            dists = d_vals_out[..., 1:] - d_vals_out[..., :-1]
            dists = torch.cat([dists, 1e10 * torch.ones(dists[..., :1].shape).to(device)], dim=-1)
            alpha_out = 1 - torch.exp(-F.softplus(sigma_out) * dists)   # use softplus instead of relu as NeuS's official repo
        
        # --------------
        # Ray Integration
        # --------------
        # [(B), N_rays, N_pts-1]
        if N_outside > 0:
            N_pts_1 = d_mid.shape[-1]
            # [(B), N_ryas, N_pts-1]
            mask_inside = (pts_mid.norm(dim=-1) <= obj_bounding_radius)
            # [(B), N_ryas, N_pts-1]
            alpha_in = opacity_alpha * mask_inside.float() + alpha_out[..., :N_pts_1] * (~mask_inside).float()
            # [(B), N_ryas, N_pts-1 + N_outside]
            opacity_alpha = torch.cat([alpha_in, alpha_out[..., N_pts_1:]], dim=-1)
            
            # [(B), N_ryas, N_pts-1, 3]
            radiance_in = radiances * mask_inside.float()[..., None] + radiance_out[..., :N_pts_1, :] * (~mask_inside).float()[..., None]
            # [(B), N_ryas, N_pts-1 + N_outside, 3]
            radiances = torch.cat([radiance_in, radiance_out[..., N_pts_1:, :]], dim=-2)
            d_final = d_vals_out
        else:
            d_final = d_mid

        # [(B), N_ryas, N_pts-1 + N_outside]
        visibility_weights = alpha_to_w(opacity_alpha)
        # [(B), N_rays]
        rgb_map = torch.sum(visibility_weights[..., None] * radiances, -2)
        # depth_map = torch.sum(visibility_weights * d_mid, -1)
        # NOTE: to get the correct depth map, the sum of weights must be 1!
        depth_map = torch.sum(visibility_weights / (visibility_weights.sum(-1, keepdim=True)+1e-10) * d_final, -1)
        acc_map = torch.sum(visibility_weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        ret_i = OrderedDict([
            ('rgb', rgb_map),           # [(B), N_rays, 3]
            ('depth_volume', depth_map),     # [(B), N_rays]
            # ('depth_surface', d_pred_out),    # [(B), N_rays]
            ('mask_volume', acc_map)            # [(B), N_rays]
        ])

        if calc_normal:
            normals_map = F.normalize(nablas, dim=-1)
            N_pts = min(visibility_weights.shape[-1], normals_map.shape[-2])
            normals_map = (normals_map[..., :N_pts, :] * visibility_weights[..., :N_pts, None]).sum(dim=-2)
            ret_i['normals_volume'] = normals_map

        if detailed_output:
            ret_i['implicit_nablas'] = nablas
            ret_i['implicit_surface'] = sdf
            ret_i['radiance'] = radiances
            ret_i['alpha'] = opacity_alpha
            ret_i['cdf'] = cdf
            ret_i['visibility_weights'] = visibility_weights
            ret_i['d_final'] = d_final
            if N_outside > 0:
                ret_i['sigma_out'] = sigma_out
                ret_i['radiance_out'] = radiance_out

        return ret_i
        
    ret = {}
    for i in tqdm(range(0, rays_o.shape[DIM_BATCHIFY], rayschunk), disable=not show_progress):
        ret_i = render_rayschunk(
            rays_o[:, i:i+rayschunk] if batched else rays_o[i:i+rayschunk],
            rays_d[:, i:i+rayschunk] if batched else rays_d[i:i+rayschunk]
        )
        for k, v in ret_i.items():
            if k not in ret:
                ret[k] = []
            ret[k].append(v)
    for k, v in ret.items():
        ret[k] = torch.cat(v, DIM_BATCHIFY)
    
    return ret['rgb'], ret['depth_volume'], ret

class SingleRenderer(nn.Module):
    def __init__(self, model: NeuS):
        super().__init__()
        self.model = model

    def forward(self, rays_o, rays_d, **kwargs):
        return volume_render(rays_o, rays_d, self.model, **kwargs)


class Trainer(nn.Module):
    def __init__(self, model: NeuS, device_ids=[0], batched=True,is_finetune = False,target_hw:list = None):
        super().__init__()
        self.model = model
        self.renderer = SingleRenderer(model)
        if len(device_ids) > 1:
            self.renderer = nn.DataParallel(self.renderer, device_ids=device_ids, dim=1 if batched else 0)
        self.device = device_ids[0]
        self.neg_texts = None
        if target_hw is None:
            target_hw = [960, 540] #full-res 4:3
        if is_finetune: #if is clip finetune, load loss here
            contrastive_loss = ContrastiveLoss()
            patchnce_loss = PatchNCELoss(target_hw).cuda()
            clip_loss = CLIPLoss()#, src_img_list=src_paths, tar_img_list=tar_paths)
            perp_loss = VGGPerceptualLoss().cuda()
            loss_dict = {'contrastive': contrastive_loss, 'patchnce': patchnce_loss,\
                 'clip': clip_loss, 'perceptual': perp_loss}
            self.loss_dict = loss_dict
            #! for disentanglement
            global FIX_MODULE
            self.model.fix_module(FIX_MODULE)

    def create_fine_neg_texts(self, args):
        path = "criteria/neg_text.txt"
        results = {}
        curr_key = 0
        with open(path, 'r') as fr:
            contents = fr.readlines()
            for item in contents:
                item = item.strip()
                if item.startswith("#"):
                    curr_key = item[1:]
                    results[curr_key] = []
                else:
                    results[curr_key].append(item.split(".")[1])
        
        all_texts = []
        remove_ids = [] 
        ttext = args.finetune.target_text.lower()
        if 'botero' in ttext or 'monalisa' in ttext or 'portrait' in ttext or 'painting' in ttext:
            remove_ids = ['portrait']
        elif 'zombie' in ttext:
            remove_ids = ['zombie']
        elif 'wolf' in ttext:
            remove_ids = ['wolf']
        elif 'pixlar' in ttext or 'disney' in ttext:
            remove_ids = ['disney']
        elif 'sketch' in ttext:
            remove_ids = ['sketch'] 

        for key in results:
            if key not in remove_ids:
            #if key in remove_ids:
                all_texts += results[key]
        return all_texts

    
    def forward(self, 
                args,
                indices,
                model_input,
                ground_truth,
                render_kwargs_train: dict,
                it: int,
                device='cuda',
                optimizer = None):
        if self.neg_texts is None:
            self.neg_texts = self.create_fine_neg_texts(args=args)

        device = "cuda"
        intrinsics = model_input["intrinsics"].to(device)
        c2w = model_input['c2w'].to(device)
        H = render_kwargs_train['H']
        W = render_kwargs_train['W']
        train_N_rays =  -1 if args.training.is_finetune else args.data.N_rays
        rays_o, rays_d, select_inds = rend_util.get_rays(
            c2w, intrinsics, H, W, N_rays=train_N_rays)
        # [B, N_rays, 3]
        target_rgb = torch.gather(ground_truth['rgb'].to(device), 1, torch.stack(3*[select_inds],-1))

        if "mask_ignore" in model_input:
            mask_ignore = torch.gather(model_input["mask_ignore"].to(device), 1, select_inds)
        else:
            mask_ignore = None
        if args.training.is_finetune:
            # rays_chunk = 2000 if args.finetune.use_eikonal else 4000
            #! Aug 14: for debug
            rays_chunk = 4800
            losses = None
            #when use eikonal, autograd will be turned on later
            with torch.no_grad():
                rgb, depth_v, extras = self.renderer(rays_o, rays_d, detailed_output=True,\
                    use_view_dirs=args.model.radiance.use_view_dirs,\
                    # require_nablas = args.finetune.use_eikonal,
                    require_nablas = args.finetune.use_eikonal or args.model.radiance.use_view_dirs,\
                    rayschunk = rays_chunk, **render_kwargs_train)

            #TODO Aug 04: add loss as per StyleNerf
            rgb.requires_grad_(True)
            losses = self.calc_style_loss(rgb,target_rgb,args,H=H)

            losses.backward()

            gradient = rgb.grad.clone().detach()                    

            optimizer.zero_grad()
            batch_size = 1200 #TODO Aug 05: hardcoded for 3090Ti
            # #! Aug 06: dump rgb for debug
            # import pickle
            # with open("rgb.pkl", "wb") as f:
            #     pickle.dump(rgb.detach().cpu().numpy(), f)
            # with open("rgb_target.pkl", "wb") as f:
            #     pickle.dump(target_rgb.detach().cpu().numpy(), f)
            #gradient: 1, N_rays, 3
            avg_eikonal_loss = 0.
            for i in range(0, gradient.shape[1], batch_size):
                rays_o_patch, rays_d_patch = rays_o[:, i:i+batch_size,:], rays_d[:, i:i+batch_size,:]
                
                rgb_pred, _, extras = self.renderer(rays_o_patch, rays_d_patch, detailed_output=True,\
                    require_nablas = args.finetune.use_eikonal or args.model.radiance.use_view_dirs,\
                         use_view_dirs=args.model.radiance.use_view_dirs, **render_kwargs_train)
                
                rgb_pred.backward(gradient[:, i:i+batch_size, :],retain_graph=args.finetune.use_eikonal)
                #! Aug 08: try only calc and backward eikonal in 2nd pass
                if args.finetune.use_eikonal:
                    eikonal_loss = self.calc_eikonal_loss(extras, args, H)
                    eikonal_loss.backward(retain_graph=False)
                    avg_eikonal_loss += eikonal_loss.detach().cpu().numpy()
                    # nablas_patch = extras['implicit_nablas'].flatten(-3, -2)
                    # nablas_patch.backward(nablas_grad[:, i:i+batch_size, :])
                    del eikonal_loss
                # free the graph by deleting the variable since we retain_graph
                del rgb_pred, extras, _
                gc.collect()
            avg_eikonal_loss /= gradient.shape[1]//batch_size
            print("\tEikonal loss: ",avg_eikonal_loss * args.finetune.w_perceptual)
            # loss_val = losses.detach().clone().cpu().numpy()
            #! Aug 15: prevent memory leak
            del gradient, rgb, avg_eikonal_loss, depth_v, rays_d, rays_o
            gc.collect()
            torch.cuda.empty_cache()

        else: #else is reconstruction of surface
            rgb, depth_v, extras = self.renderer(rays_o, rays_d, detailed_output=True, **render_kwargs_train)

            # [B, N_rays, N_pts, 3]
            nablas: torch.Tensor = extras['implicit_nablas']
            # [B, N_rays, N_pts]
            nablas_norm = torch.norm(nablas, dim=-1)
            # [B, N_rays]
            mask_volume: torch.Tensor = extras['mask_volume']
            # NOTE: when predicted mask is close to 1 but GT is 0, exploding gradient.
            # mask_volume = torch.clamp(mask_volume, 1e-10, 1-1e-10)
            mask_volume = torch.clamp(mask_volume, 1e-3, 1-1e-3)
            extras['mask_volume_clipped'] = mask_volume

            losses = OrderedDict()

            # [B, N_rays, 3]
            losses['loss_img'] = F.l1_loss(rgb, target_rgb, reduction='none')
            # [B, N_rays, N_pts]
            losses['loss_eikonal'] = args.training.w_eikonal * F.mse_loss(nablas_norm, nablas_norm.new_ones(nablas_norm.shape), reduction='mean')

            if args.training.with_mask:
                # [B, N_rays]
                target_mask = torch.gather(model_input["object_mask"].to(device), 1, select_inds)
                losses['loss_mask'] = args.training.w_mask * F.binary_cross_entropy(mask_volume, target_mask.float(), reduction='mean')
                if mask_ignore is not None:
                    target_mask = torch.logical_and(target_mask, mask_ignore)
                # [N_masked, 3]
                losses['loss_img'] = (losses['loss_img'] * target_mask[..., None].float()).sum() / (target_mask.sum()+1e-10)
            else:
                if mask_ignore is not None:
                    losses['loss_img'] = (losses['loss_img'] * mask_ignore[..., None].float()).sum() / (mask_ignore.sum() + 1e-10)
                else:
                    losses['loss_img'] = losses['loss_img'].mean()

            loss = 0
            for k, v in losses.items():
                loss += losses[k]

            losses['total'] = loss
        if args.training.is_finetune:
            extras = {}
            # losses = loss_val
        else:
            extras['implicit_nablas_norm'] = nablas_norm
            extras['scalars'] = {'1/s': 1./self.model.forward_s().data}
            extras['select_inds'] = select_inds
        
        return OrderedDict(
            [('losses', losses),
             ('extras', extras)])
    def calc_style_loss(self,rgb:torch.Tensor,rgb_gt:torch.Tensor,args, H = 480):
        """
        Calculate CLIP-driven style losses

        Input
        -----
        rgb: torch.Tensor, [B, H*W, 3]
        rgb_gt: torch.Tensor, [B, H*W,3 ]
        """
        loss = 0.0
        rgb_pred = rearrange(rgb, "B (H W) C -> B C H W", H=H)
        rgb_gt = rearrange(rgb_gt, "B (H W) C -> B C H W", H=H)
        s_text = args.finetune.src_text
        t_text = args.finetune.target_text
        #direct clip loss
        dir_clip_loss = self.loss_dict["clip"](rgb_gt,s_text,rgb_pred,t_text)
        loss = loss + dir_clip_loss * args.finetune.w_clip
        print("Directional CLIP loss:", dir_clip_loss.data.detach().cpu().numpy() * args.finetune.w_clip)
        #persptual
        perp_loss = self.loss_dict["perceptual"](rgb_pred, rgb_gt)
        loss = loss + perp_loss * args.finetune.w_perceptual
        print("Perceptual loss:", perp_loss.data.detach().cpu().numpy() * args.finetune.w_perceptual)
        #Global contrastive
        s_text_list = self.neg_texts#[:100] # More negtive samples 
        s_text = random.choice(s_text_list)
        loss_contrastive = self.loss_dict["contrastive"](rgb_gt, s_text, rgb_pred, t_text)
        loss = loss + loss_contrastive * args.finetune.w_contrastive

        #local contrastive
        #stexts = ['Photo', 'Human', 'Human face', 'Real face']
        neg_counts = 8
        s_text_list = random.sample(self.neg_texts, neg_counts)
        is_full_res = args.data.downscale == 1
        loss_patchnce = self.loss_dict["patchnce"](s_text_list, rgb_pred, t_text,is_full_res)
        loss = loss + loss_patchnce * args.finetune.w_patchnce

        return loss
        
    def calc_eikonal_loss(self, extras, args, H):
        #! Sep 8 modified to check impact of eikonal loss
        # [B, N_rays, N_pts, 3]            
        nablas: torch.Tensor = extras['implicit_nablas']
        # [B, N_rays, ]
        # #---------- OPTION1: just flatten and use all nablas
        nablas = nablas.flatten(-3, -2)
        #---------- OPTION2: using only one point each ray: this may be what the paper suggests.
        # @ VolSDF section 3.5, "combine a SINGLE random uniform space point and a SINGLE point from \mathcal{S} for each pixel"
#         _, _ind = extras['visibility_weights'][..., :nablas.shape[-2]].max(dim=-1)
#         nablas = torch.gather(nablas, dim=-2, index=_ind[..., None, None].repeat([*(len(nablas.shape)-1)*[1], 3]))
        
#         eik_bounding_box = args.model.obj_bounding_radius
#         eikonal_points = torch.empty_like(nablas).uniform_(-eik_bounding_box, eik_bounding_box).to(nablas.device)
#         _, nablas_eik, _ = self.model.implicit_surface.forward_with_nablas(eikonal_points)
#         # [B, N_rays, 2, 3]     
#         nablas = torch.cat([nablas, nablas_eik], dim=-2)

        # [B, N_rays, N_pts]
        nablas_norm = torch.norm(nablas, dim=-1)

        eikonal_loss = args.finetune.w_eikonal * F.mse_loss(nablas_norm, nablas_norm.new_ones(nablas_norm.shape), reduction='mean')
        # print("\tEikonal loss: ", eikonal_loss.data.detach().cpu().numpy())
        return eikonal_loss


def get_model(args,render_target):
    
    if not args.training.with_mask:
        assert 'N_outside' in args.model.keys() and args.model.N_outside > 0, \
            "Please specify a positive model:N_outside for neus with nerf++"
    
    model_config = {
        'obj_bounding_radius':  args.model.obj_bounding_radius,
        'W_geo_feat':       args.model.setdefault('W_geometry_feature', 256),
        'use_outside_nerf': not args.training.with_mask,
        'speed_factor':     args.training.setdefault('speed_factor', 1.0),
        'variance_init':    args.model.setdefault('variance_init', 0.05)
    }
    
    surface_cfg = {
        'use_siren':    args.model.surface.setdefault('use_siren', args.model.setdefault('use_siren', False)),
        'embed_multires': args.model.surface.setdefault('embed_multires', 6),
        'radius_init':  args.model.surface.setdefault('radius_init', 1.0),
        'geometric_init': args.model.surface.setdefault('geometric_init', True),
        'D': args.model.surface.setdefault('D', 8),
        'W': args.model.surface.setdefault('W', 256),
        'skips': args.model.surface.setdefault('skips', [4]),
    }
        
    radiance_cfg = {
        'use_siren':    args.model.radiance.setdefault('use_siren', args.model.setdefault('use_siren', False)),
        'embed_multires': args.model.radiance.setdefault('embed_multires', -1),
        'embed_multires_view': args.model.radiance.setdefault('embed_multires_view', -1),
        'use_view_dirs': args.model.radiance.setdefault('use_view_dirs', True),
        'D': args.model.radiance.setdefault('D', 4),
        'W': args.model.radiance.setdefault('W', 256),
        'skips': args.model.radiance.setdefault('skips', []),
    }
    
    model_config['surface_cfg'] = surface_cfg
    model_config['radiance_cfg'] = radiance_cfg
    
    model = NeuS(**model_config)
    
    ## render kwargs
    render_kwargs_train = {
        # upsample config
        'upsample_algo':    args.model.setdefault('upsample_algo', 'official_solution'),    # [official_solution, direct_more, direct_use]
        'N_nograd_samples': args.model.setdefault('N_nograd_samples', 2048),
        'N_upsample_iters': args.model.setdefault('N_upsample_iters', 4),
        
        'N_outside': args.model.setdefault('N_outside', 0),
        'obj_bounding_radius': args.data.setdefault('obj_bounding_radius', 1.0),
        'batched': args.data.batch_size is not None,
        'perturb': args.model.setdefault('perturb', True),   # config whether do stratified sampling
        'white_bkgd': args.model.setdefault('white_bkgd', False),
    }
    render_kwargs_test = copy.deepcopy(render_kwargs_train)
    render_kwargs_test['rayschunk'] = args.data.val_rayschunk
    render_kwargs_test['perturb'] = False
    trainer = Trainer(model, args.device_ids, batched=render_kwargs_train['batched'],is_finetune = args.training.is_finetune,target_hw=render_target)
    
    return model, trainer, render_kwargs_train, render_kwargs_test, trainer.renderer
