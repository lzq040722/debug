import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import importlib
from tqdm import tqdm
import trimesh
import cv2
import os

import genesis as gs
import taichi as ti
import importlib

PRESET_Z_VALUE = 0.0 # 10.0


def pt3d_to_gs(xyz, no_z_offset=False):
    z_offset = 0.0 if no_z_offset else PRESET_Z_VALUE
    # xyz: [..., 3]
    # transform xyz coordinates to genesis coordinates
    # pt3d: x - left, y - up, z - forward
    # genesis: x - right, y - forward, z - up
    if isinstance(xyz, torch.Tensor):
        xyz_new = xyz.clone()
        xyz_new[..., 0] = -1 * xyz[..., 0]
        xyz_new[..., 1] = xyz[..., 2]
        xyz_new[..., 2] = xyz[..., 1]
    elif isinstance(xyz, np.ndarray):
        xyz_new = xyz.copy()
        xyz_new[..., 0] = -1 * xyz[..., 0]
        xyz_new[..., 1] = xyz[..., 2]
        xyz_new[..., 2] = xyz[..., 1]
    else:
        raise ValueError(f"Input type {type(xyz)} is not supported")

    xyz_new[..., 2] += z_offset
    return xyz_new


def gs_to_pt3d(xyz, no_z_offset=False):
    z_offset = 0.0 if no_z_offset else PRESET_Z_VALUE
    # xyz: [..., 3]
    # transform xyz coordinates to pt3d coordinates
    # pt3d: x - left, y - up, z - forward
    # genesis: x - right, y - forward, z - up
    if isinstance(xyz, torch.Tensor):
        xyz_new = xyz.clone()
        xyz_new[..., 0] = -1 * xyz[..., 0]
        xyz_new[..., 1] = xyz[..., 2]
        xyz_new[..., 2] = xyz[..., 1]
    elif isinstance(xyz, np.ndarray):
        xyz_new = xyz.copy()
        xyz_new[..., 0] = -1 * xyz[..., 0]
        xyz_new[..., 1] = xyz[..., 2]
        xyz_new[..., 2] = xyz[..., 1]
    else:
        raise ValueError(f"Input type {type(xyz)} is not supported")

    xyz_new[..., 1] -= z_offset
    return xyz_new


class Simulator(nn.Module):
    def __init__(
        self,
        config,
        obj_gaussians,  # a list of object gaussians
        env_gaussians,
        device="cuda",
        delta_time=0.1,
        save_dir=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.config = config
        self.device = device
        self.save_dir = save_dir
        self.len_obj = len(obj_gaussians)
        self.material_types = config['material_types']

        self.force_function_module = importlib.import_module('simulator.genesis_functions')
        if 'force_function_name' in config:
            self.force_function = getattr(self.force_function_module, config['force_function_name'])
        else:
            self.force_function = None

        gs.init(
            seed=config["seed"],
            precision="32",
            backend=gs.gs_backend.gpu,
            logging_level="warning"
        )

        obj_min = (torch.ones(3) * 1000.0).to(self.device)
        obj_max = (torch.ones(3) * (-1000.0)).to(self.device)

        self.obj_infos = dict()

        for idx in range(self.len_obj):
            obj_gaussian = obj_gaussians[idx]
            obj_xyz = pt3d_to_gs(obj_gaussian["xyz"])

            self.obj_infos.update({
                idx: [obj_xyz.min(dim=0)[0], obj_xyz.max(dim=0)[0]]
            })

            obj_min = torch.where(
                obj_min > obj_xyz.min(dim=0)[0],
                obj_xyz.min(dim=0)[0],
                obj_min,
            )
            obj_max = torch.where(
                obj_max < obj_xyz.max(dim=0)[0],
                obj_xyz.max(dim=0)[0],
                obj_max,
            )
        
        # enlarge object range for mpm simulation, 
        # in genesis we need to use a slightly larger region than original object range
        # because the auto-particle generation will generate particles outside the object range
        obj_range = obj_max - obj_min
        obj_valid_min = obj_min - obj_range * 1
        obj_valid_max = obj_max + obj_range * 1

        # 
        env_xyz = pt3d_to_gs(env_gaussians["xyz"])

        # generic environment bounds
        env_xyz = self.env_process(env_xyz, obj_valid_min, obj_valid_max, obj_min, obj_max)
        plane_point = env_xyz[env_xyz[:, 2].argmax()].cpu().numpy()
        if obj_valid_min[2] >= plane_point[2]:
            obj_valid_max[2] = obj_valid_max[2] + (obj_valid_min[2] - plane_point[2])
            obj_valid_min[2] = plane_point[2] - (obj_valid_min[2] - plane_point[2])
        self.obj_valid_min = obj_valid_min.cpu().numpy()
        self.obj_valid_max = obj_valid_max.cpu().numpy()

        self.dt = delta_time

        pbd_gravity = (0, 0, 1.8) if 'pbd_gas' in self.material_types else None
        mpm_gravity = None
        
        # a bit confused, normally we use same particle_size for both, or just set both as 0.01, but you can also use different particle_size for mpm and pbd
        if 'mpm_particle_size' in self.config:
            mpm_particle_size = self.config['mpm_particle_size']
        else:
            mpm_particle_size = 0.01
        
        if 'pbd_particle_size' in self.config:
            pbd_particle_size = self.config['pbd_particle_size']
        else:
            pbd_particle_size = 0.01

        self.scene = gs.Scene(
            sim_options = gs.options.SimOptions(
                dt=self.dt,
                gravity=(0, 0, -9.8),
                substeps=10 if 'substeps' not in self.config else self.config['substeps'],
            ),
            show_viewer=False,
            vis_options = gs.options.VisOptions(
                show_world_frame = True, # visualize the coordinate frame of `world` at its origin
                world_frame_size = 1.0, # length of the world frame in meter
                show_link_frame  = False, # do not visualize coordinate frames of entity links
                show_cameras     = False, # do not visualize mesh and frustum of the cameras added
                plane_reflection = True, # turn on plane reflection
                ambient_light    = (0.1, 0.1, 0.1), # ambient light setting
            ),
            renderer = gs.renderers.Rasterizer(),
            mpm_options = gs.options.MPMOptions(
                lower_bound = tuple(self.obj_valid_min),
                upper_bound = tuple(self.obj_valid_max),
                grid_density = 64 if 'MPM_grid_density' not in self.config else self.config['MPM_grid_density'],
                particle_size = mpm_particle_size if 'particle_size' not in self.config else self.config['particle_size'],
                gravity = mpm_gravity,
            ),
            pbd_options = gs.options.PBDOptions(
                lower_bound = tuple(self.obj_valid_min),
                upper_bound = tuple(self.obj_valid_max),
                particle_size = pbd_particle_size if 'particle_size' not in self.config else self.config['particle_size'],
                gravity = pbd_gravity,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                enable_collision=True,
                enable_self_collision=False,
                contact_resolve_time=0.02,  # avoid the rigid contact solver being too stiff otherwise will cause large impulse (especially we have small dt for rigid solver)
            ),
            coupler_options=gs.options.CouplerOptions(mpm_pbd=True),
        )

        self.env = None
        
        # TODO: Genesis can't deal with pbd fixed particle for now
        # TODO: so if rigid is envolved, use a plane as environment
        if 'rigid' not in self.material_types:
            env_xyz_np = env_xyz.cpu().numpy()
            env_xyz_np_path = os.path.join(self.save_dir, "env_xyz.txt")
            np.savetxt(env_xyz_np_path, env_xyz_np)

            self.env = self.scene.add_entity(
                material = gs.materials.PBD.Particle(),
                morph = gs.morphs.Points(
                    points_path=env_xyz_np_path,
                ),
                surface = gs.surfaces.Default(
                    color = (0.8, 0.8, 0.8, 1.0),
                    vis_mode = "particle",
                )
            )
            env_particle_num = self.env.n_particles
        
        if self.env is None:
            # Single rigid object (e.g. venice): support plane at object base
            lowest_point = self.obj_infos[0][0].cpu().numpy().reshape(3)
            self.scene_plane_2 = self.scene.add_entity(
                morph=gs.morphs.Plane(pos=(lowest_point[0], lowest_point[1], lowest_point[2]), normal=(0, 0, 1)),
            )

        self.objs = []
        self.obj_gaussians = obj_gaussians
        self.num_vertices = []
        self.verts_closest_indices = []
        self.particles_pos_start_in_gs = []
        self.random_colors = np.random.rand(self.len_obj, 3)

        self.rigid_mesh_vertices = dict()
        self.pbd_cloth_fix = dict()

        self.init_meshes = []
        self.init_meshes_path = []
        self.init_translations = []

        for idx in range(self.len_obj):
            obj_gaussian = obj_gaussians[idx]
            obj_mesh_path = obj_gaussian["mesh_path"]
            obj_mesh_new_path = obj_mesh_path[:-4] + "_gscoord.obj"
            tmp_mesh = trimesh.load(obj_mesh_path, process=False)
            tmp_mesh.vertices = pt3d_to_gs(np.array(tmp_mesh.vertices))
            tmp_mesh.export(obj_mesh_new_path)
            os.remove(obj_mesh_path)

            translation = obj_gaussian["translation"]
            translation = np.array(translation).reshape(1, 3)
            translation = pt3d_to_gs(translation, no_z_offset=True).reshape(3)
            translation = tuple(translation)

            self.init_meshes.append(tmp_mesh)
            self.init_meshes_path.append(obj_mesh_new_path)
            self.init_translations.append(translation)
        
        for idx in range(self.len_obj):
            obj_gaussian = obj_gaussians[idx]
            num_verts = obj_gaussian["xyz"].shape[0]
            tmp_mesh = self.init_meshes[idx]
            obj_mesh_new_path = self.init_meshes_path[idx]
            translation = self.init_translations[idx]

            obj_material = None
            obj_material_type = self.material_types[idx]

            if obj_material_type == "rigid":
                obj_material = gs.materials.Rigid(
                    friction = None if 'rigid_friction' not in self.config else self.config['rigid_friction'],
                    coup_friction = 0.1 if 'rigid_coup_friction' not in self.config else self.config['rigid_coup_friction'],
                    coup_softness = 0.002 if 'rigid_coup_softness' not in self.config else self.config['rigid_coup_softness'],
                )
                obj_vis_mode = "visual"
                self.rigid_mesh_vertices.update({idx: None})

            elif obj_material_type == "pbd_liquid":
                obj_material = gs.materials.PBD.Liquid(
                    rho = self.config['pbd_rho'],
                    density_relaxation = 0.2 if 'pbd_density_relaxation' not in self.config else self.config['pbd_density_relaxation'],
                    viscosity_relaxation = 0.01 if 'pbd_viscosity_relaxation' not in self.config else self.config['pbd_viscosity_relaxation'],
                )
                obj_vis_mode = "particle"
            
            elif obj_material_type == "pbd_gas":
                obj_material = gs.materials.PBD.Liquid(
                    rho = self.config['pbd_rho'],
                    density_relaxation = 0.2 if 'pbd_density_relaxation' not in self.config else self.config['pbd_density_relaxation'],
                    viscosity_relaxation = 0.01 if 'pbd_viscosity_relaxation' not in self.config else self.config['pbd_viscosity_relaxation'],
                )
                obj_vis_mode = "particle"
            
            elif obj_material_type == "pbd_cloth":
                obj_material = gs.materials.PBD.Cloth(
                    static_friction=0.15 if 'pbd_static_friction' not in self.config else self.config['pbd_static_friction'],
                    kinetic_friction=0.15 if 'pbd_kinetic_friction' not in self.config else self.config['pbd_kinetic_friction'],
                    stretch_compliance=1e-7 if 'pbd_stretch_compliance' not in self.config else self.config['pbd_stretch_compliance'],
                    bending_compliance=1e-5 if 'pbd_bending_compliance' not in self.config else self.config['pbd_bending_compliance'],
                    stretch_relaxation=0.3 if 'pbd_stretch_relaxation' not in self.config else self.config['pbd_stretch_relaxation'],
                    bending_relaxation=0.1 if 'pbd_bending_relaxation' not in self.config else self.config['pbd_bending_relaxation'],
                    air_resistance=1e-3 if 'pbd_air_resistance' not in self.config else self.config['pbd_air_resistance'],
                    # rho = self.config['pbd_rho'],
                    # stretch_relaxation = 0.3,
                    # bending_relaxation = 0.1,
                )
                obj_vis_mode = "particle"

            elif obj_material_type == "pbd_elastic":
                obj_material = gs.materials.PBD.Elastic(
                    rho = self.config['pbd_rho'],
                    static_friction = 0.15 if 'pbd_static_friction' not in self.config else self.config['pbd_static_friction'],
                    kinetic_friction = 0.15 if 'pbd_kinetic_friction' not in self.config else self.config['pbd_kinetic_friction'],
                )
                obj_vis_mode = "particle"

            elif obj_material_type == "pbd_particle":
                obj_material = gs.materials.PBD.Particle()
                obj_vis_mode = "particle"
            
            elif obj_material_type == "mpm_elastic":
                obj_material = gs.materials.MPM.Elastic(
                    E = self.config['MPM_E'],
                    nu = self.config['MPM_nu'],
                    rho = self.config['MPM_rho'],
                    sampler = 'pbs' if 'MPM_sampler' not in self.config else self.config['MPM_sampler'],
                )
                obj_vis_mode = "particle"

            elif obj_material_type == "mpm_elastoplastic":
                obj_material = gs.materials.MPM.ElastoPlastic(
                    E = self.config['MPM_E'],
                    nu = self.config['MPM_nu'],
                    rho = self.config['MPM_rho'],
                    yield_lower = 2.5e-2 if 'MPM_yield_lower' not in self.config else self.config['MPM_yield_lower'],
                    yield_higher = 4.5e-3 if 'MPM_yield_higher' not in self.config else self.config['MPM_yield_higher'],
                    von_mises_yield_stress = 10000.0 if 'MPM_von_mises_yield_stress' not in self.config else self.config['MPM_von_mises_yield_stress'],
                )
                obj_vis_mode = "particle"

            elif obj_material_type == "mpm_sand":
                obj_material = gs.materials.MPM.Sand(
                    E = self.config['MPM_E'],
                    nu = self.config['MPM_nu'],
                    rho = self.config['MPM_rho'],
                    friction_angle = 45 if 'MPM_friction_angle' not in self.config else self.config['MPM_friction_angle'],
                )
                obj_vis_mode = "particle"
            
            elif obj_material_type == "mpm_liquid":
                obj_material = gs.materials.MPM.Liquid(
                    E = self.config['MPM_E'],
                    nu = self.config['MPM_nu'],
                    rho = self.config['MPM_rho'],
                    viscous = False if 'MPM_viscous' not in self.config else self.config['MPM_viscous'],
                )
                obj_vis_mode = "particle"
            
            elif obj_material_type == "mpm_snow":
                obj_material = gs.materials.MPM.Snow(
                    E = self.config['MPM_E'],
                    nu = self.config['MPM_nu'],
                    rho = self.config['MPM_rho'],
                )
                obj_vis_mode = "particle"
            
            elif obj_material_type == "fem_elastic":
                # TODO: fix vertices
                obj_material = gs.materials.FEM.Elastic(
                    E = self.config['MPM_E'],
                    nu = self.config['MPM_nu'],
                    rho = self.config['MPM_rho'],
                )
                obj_vis_mode = "visual"

            else:
                raise ValueError(f"Object {idx} with material type {obj_material_type} is not supported")

            obj = self.scene.add_entity(
                material=obj_material,
                morph=gs.morphs.Mesh(
                    file=obj_mesh_new_path,
                    scale=1.0,
                    pos=translation,
                    euler=(0.0, 0.0, 0.0),
                ),
                surface=gs.surfaces.Default(
                    color=(self.random_colors[idx][0], self.random_colors[idx][1], self.random_colors[idx][2], 1.0),
                    vis_mode=obj_vis_mode,
                ),
            )

            # if idx in self.config['rigid_id_list']:
            #     # a hack way as our rigid object is a complete mesh without links
            #     # self.particles_pos_start_in_gs.append(obj._links[0]._vgeoms[0]._init_vverts)
            #     # sim_particles = torch.tensor(obj._links[0]._vgeoms[0]._init_vverts).to(self.device)
            #     self.particles_pos_start_in_gs.append(obj._links[0]._geoms[0]._init_verts)
            #     sim_particles = torch.tensor(obj._links[0]._geoms[0]._init_verts).to(self.device)

            # elif idx in self.config['elastoplastic_id_list'] or idx in self.config['elastic_id_list'] or idx in self.config['granular_id_list']:
            #     self.particles_pos_start_in_gs.append(obj.init_particles)
            #     sim_particles = torch.tensor(obj.init_particles).to(self.device)
            
            if obj_material_type in ["pbd_gas"]:
                mesh_vertices = torch.tensor(np.array(tmp_mesh.vertices)).to(self.device)
                mesh_vertices_translations = torch.tensor(self.init_translations[idx]).to(self.device)
                mesh_vertices = mesh_vertices + mesh_vertices_translations
                # in the world space
                sim_particles = torch.tensor(obj.init_particles).to(self.device)

                # Split vertices into K chunks for memory efficiency
                K = 8192  # Adjust K based on available memory
                if self.config['example_name'] == "smoke_can":
                    K = 2048
                num_closest = 5
                vertex_chunks = torch.split(mesh_vertices, K)
                closest_indices = []

                for chunk in tqdm(vertex_chunks):
                    # Calculate pairwise distances between chunk and all particles
                    # Using broadcasting to avoid memory issues
                    # Shape: [K, 1, 3] - [1, N, 3] -> [K, N, 3] -> [K, N]
                    distances = torch.norm(
                        chunk.unsqueeze(1) - sim_particles.unsqueeze(0),
                        dim=2
                    )
                    # Get top num_closest indices of closest particles for this chunk
                    chunk_closest = torch.topk(distances, k=num_closest, dim=1, largest=False)[1]   # [K, num_closest]
                    del distances
                    closest_indices.append(chunk_closest)

                # Combine all closest indices
                closest_indices = torch.cat(closest_indices)
            
            else:
                closest_indices = None

            self.objs.append(obj)
            self.num_vertices.append(num_verts)
            self.verts_closest_indices.append(closest_indices)
        
        self.cam = self.scene.add_camera(
            res = (512, 512),
            pos = (0, 0, PRESET_Z_VALUE),
            lookat = (0, 5, PRESET_Z_VALUE),
            fov = 40,
            GUI = False,
        )

        self.emitter = None

        for idx in range(self.len_obj):
            obj_material_type = self.material_types[idx]
            if obj_material_type == "mpm_elastic":
                if self.config['example_name'] == "demo_7":
                    wind_force_direction = (1, 0.2, 0.3)
                    wind_force_direction = np.array(wind_force_direction) / np.linalg.norm(np.array(wind_force_direction))

                    vortex_center = np.array([
                        (self.init_translations[1][0] + self.init_translations[2][0]).item() / 2,
                        (self.init_translations[1][1] + self.init_translations[2][1]).item() / 2,
                        (self.init_translations[1][2] + self.init_translations[2][2]).item() / 2,
                    ])
                    center_ti = ti.Vector(vortex_center, dt=gs.ti_float)
                    @ti.func
                    def force_func(pos, vel, t, i):
                        strength_perpendicular = 2000.0
                        strength_radial = 100.0
                        # strength_perpendicular = 50 # 2000.0
                        # strength_radial = 400 # 100.0
                        strength = 12.
                        damping = 0.0
                        acceleration = ti.Vector.zero(gs.ti_float, 3)
                        direction = (0.0, 1.0, 0.0)
                        direction_ti = ti.Vector(direction, dt=gs.ti_float)

                        if t > 400 * self.dt:
                            direction_ti = ti.Vector(wind_force_direction, dt=gs.ti_float)
                            dist = (pos - center_ti).cross(direction_ti).norm()
                            acc = direction_ti * strength
                            acceleration = acc
                        
                        else:
                            spacings = 150
                            if int((t / self.dt) / spacings) % 2 == 0:
                                direction_ti = ti.Vector([0.0, 0.0, 1.0], dt=gs.ti_float)
                            else:
                                direction_ti = ti.Vector([0.0, 0.0, -1.0], dt=gs.ti_float)

                            pos_centered = pos - center_ti
                            # Project onto direction vector to get parallel component
                            proj = pos_centered.dot(direction_ti) * direction_ti
                            # Subtract parallel component to get perpendicular component
                            relative_pos = pos_centered - proj

                            radius = relative_pos.norm()
                            perpendicular = direction_ti.cross(relative_pos)
                            radial = -relative_pos

                            falloff = 1.0

                            acceleration = falloff * (strength_perpendicular * perpendicular + strength_radial * radial)

                            acceleration -= damping * vel

                        return acceleration

                    force_field = gs.force_fields.Custom(force_func)
                    force_field.activate()
                    self.scene.add_force_field(
                        force_field = force_field
                    )

                elif self.config['example_name'] == "tree_wind":
                    wind_force_center = (
                        self.obj_valid_min[0].item(),
                        (self.obj_valid_min[1] + self.obj_valid_max[1]).item() / 2,
                        (self.obj_valid_min[2] + self.obj_valid_max[2]).item() / 2,
                    )
                    wind_force_radius = (self.obj_valid_max[0] - self.obj_valid_min[0]).item() * 10.
                    direction_np = (1, 0, 0)
                    direction_np_neg = (-1, 0, 0)
                    direction_np = direction_np / np.linalg.norm(direction_np)
                    direction_np_neg = direction_np_neg / np.linalg.norm(direction_np_neg)
                    dt = delta_time

                    @ti.func
                    def force_func_tree_wind(pos, vel, t, i):
                        center = ti.Vector(wind_force_center, dt=gs.ti_float)
                        strength = 60.
                        spacing = 200.0

                        direction = ti.Vector(direction_np, dt=gs.ti_float)
                        direction_neg = ti.Vector(direction_np_neg, dt=gs.ti_float)
                        spacing = spacing * dt

                        acc_ti = ti.Vector(direction_np * strength, dt=gs.ti_float)
                        acc_ti_neg = ti.Vector(direction_np_neg * strength, dt=gs.ti_float)
                        dist = (pos - center).cross(direction).norm()
                        acc = ti.Vector.zero(gs.ti_float, 3)

                        if dist > wind_force_radius:
                            acc = ti.Vector.zero(gs.ti_float, 3)
                        else:
                            acc = acc_ti
                        
                        if (t // spacing) % 2 == 0:
                            acc = acc_ti
                        else:
                            acc = acc_ti_neg
                        
                        return acc

                    force_field = gs.force_fields.Custom(force_func_tree_wind)
                    force_field.activate()
                    self.scene.add_force_field(
                        force_field = force_field
                    )

            elif obj_material_type == "pbd_gas":
                if self.config['example_name'] == "smoke_interaction_2":
                    wind_force_center = (
                        self.obj_valid_min[0].item(),
                        (self.obj_valid_min[1] + self.obj_valid_max[1]).item() / 2,
                        (self.obj_valid_min[2] + self.obj_valid_max[2]).item() / 2,
                    )
                    wind_force_radius = (self.obj_valid_max[0] - self.obj_valid_min[0]).item() * 10.
                    wind_force_direction = (
                        1,
                        0,
                        0
                    )
                    force_field = gs.force_fields.Wind(
                        direction = wind_force_direction,
                        strength = 1.0,
                        radius = wind_force_radius,
                        center = wind_force_center,
                    )
                    force_field.activate()
                    self.scene.add_force_field(
                        force_field = force_field
                    )

                    obj_range_x_min = self.obj_infos[idx][0][0]
                    obj_range_x_max = self.obj_infos[idx][1][0]
                    obj_range_y_min = self.obj_infos[idx][0][1]
                    obj_range_y_max = self.obj_infos[idx][1][1]
                    obj_range_z_min = self.obj_infos[idx][0][2]
                    obj_range_z_max = self.obj_infos[idx][1][2]

                    emitter_N_x = 20
                    emitter_N_y = 1
                    # Sample N points in x and y ranges
                    x_points = torch.linspace(obj_range_x_min, obj_range_x_max, emitter_N_x)
                    y_points = torch.linspace(obj_range_y_min, obj_range_y_max, emitter_N_y)
                    
                    # Create meshgrid of x,y coordinates
                    X, Y = torch.meshgrid(x_points, y_points, indexing='ij')
                    
                    # Create array of z coordinates at obj_range_z_min
                    Z = torch.full_like(X, obj_range_z_min)
                    
                    # Stack into Nx3 array of xyz coordinates
                    emitter_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1).to(self.device)
                    smoke_pts = self.objs[0].vmesh.verts
                    smoke_pts = torch.tensor(smoke_pts).to(self.device)
                    # Compute distances between emitter points and smoke points
                    emitter_points_expanded = emitter_points.unsqueeze(1)  # [N, 1, 3]
                    smoke_pts_expanded = smoke_pts.unsqueeze(0)  # [1, M, 3]
                    
                    # Calculate pairwise distances
                    distances = torch.sum((emitter_points_expanded - smoke_pts_expanded) ** 2, dim=-1)  # [N, M]
                    
                    # Get indices of closest smoke points for each emitter point
                    closest_indices = torch.argmin(distances, dim=1)  # [N]
                    
                    # Use these indices to get corresponding features
                    emitter_features_dc = self.obj_gaussians[0]["features_dc"][closest_indices]   # [N, 1, 3]

                    self.emitter = []

                    for emitter_id in range(emitter_points.shape[0]):
                        emitter = self.scene.add_emitter(
                            material = gs.materials.PBD.Liquid(
                                rho = self.config['pbd_rho'],
                                density_relaxation = self.config['pbd_density_relaxation'],
                                viscosity_relaxation = self.config['pbd_viscosity_relaxation'],
                            ),
                            max_particles = 800000,
                            surface = gs.surfaces.Rough(
                                color = (0.0, 0.9, 0.4, 1.0),
                            ),
                        )
                        self.emitter.append(emitter)
                    self.emitter_pos = emitter_points.cpu().numpy()
                    self.emitter_features_dc = emitter_features_dc
                    self.emitter_dir = np.array([0, 0, 1])
                    # TODO: debug this and comment this when emitting is finished
                    # self.emitter = None
                
                elif self.config['example_name'] == "demo_6":
                    wind_force_center = (
                        self.init_translations[-2][0],
                        (self.obj_valid_min[1] + self.obj_valid_max[1]).item() / 2,
                        (self.obj_valid_min[2] + self.obj_valid_max[2]).item() / 2
                    )
                    wind_force_radius = (self.init_translations[-1][0] - self.init_translations[-2][0]) * 1.0
                    if self.config['runs_dir'].endswith("demo_6_mushroom_swinging_back"):
                        wind_force_direction = (
                            1, 0.05, 0
                        )
                    else:
                        wind_force_direction = (
                            1, 0, 0
                        )
                    force_field = gs.force_fields.Wind(
                        direction = wind_force_direction,
                        strength = 50,
                        radius = wind_force_radius,
                        center = wind_force_center,
                    )
                    force_field.activate()
                    self.scene.add_force_field(
                        force_field = force_field
                    )
                
                elif self.config['example_name'] == "smoke_can":
                    wind_lowest = self.obj_valid_min[2] + (self.obj_valid_max[2] - self.obj_valid_min[2]) * 0.48
                    wind_highest = self.obj_valid_min[2] + (self.obj_valid_max[2] - self.obj_valid_min[2]) * 0.55

                    @ti.func
                    def force_func_smoke_can(pos, vel, t, i):
                        strength = 4
                        direction = ti.Vector([-1, 0, 0], dt=gs.ti_float)

                        # smaller than 1
                        acc = ti.Vector.zero(gs.ti_float, 3)
                        if pos[2] > wind_lowest and pos[2] < wind_highest:
                            scaler = (pos[2] - wind_lowest) / (wind_highest - wind_lowest)
                            scaler = ti.exp(scaler ** 4)
                            acc = direction * strength * scaler
                        else:
                            acc = ti.Vector.zero(gs.ti_float, 3)
                        return acc

                    force_field = gs.force_fields.Custom(force_func_smoke_can)
                    force_field.activate()
                    self.scene.add_force_field(
                        force_field = force_field
                    )

            elif obj_material_type == "rigid":
                if self.config['example_name'] == "demo_8":
                    particles = np.array(self.objs[0].vgeoms[0].init_vverts)
                    particles = particles + self.init_translations[0]
                    pos_x = particles[:,0]
                    pos_y = particles[:,1]
                    pos_z = particles[:,2]

                    start_pos = np.array([
                        pos_x.min()+(pos_x.max()-pos_x.min()) * 0.7, 
                        pos_y.min()+(pos_y.max()-pos_y.min()) * 0.5, 
                        pos_z.max()+(pos_z.max()-pos_z.min()) * 0.8
                    ])
                    direction = np.array([
                        pos_x.min()+(pos_x.max()-pos_x.min()) * 0.5, 
                        pos_y.min()+(pos_y.max()-pos_y.min()) * 0.5, 
                        pos_z.min()+(pos_z.max()-pos_z.min()) * 0.5]) - start_pos
                    self.emitter_pos = start_pos
                    self.emitter_dir = direction

                    self.emitter = self.scene.add_emitter(
                        material=gs.materials.MPM.Liquid(
                            E = self.config['MPM_E'],
                            nu = self.config['MPM_nu'],
                            rho = self.config['MPM_rho'],
                            viscous = self.config['MPM_viscous'],
                        ),
                        max_particles=800000,
                        surface=gs.surfaces.Rough(
                            color=(0.0, 0.9, 0.4, 1.0),
                        ),
                    )
            
            elif obj_material_type == "mpm_snow":
                if self.config['example_name'] == "snow_statue":
                    # wind_force_center = (
                    #     self.obj_valid_min[0].item(),
                    #     self.obj_valid_min[1].item(),
                    #     self.obj_valid_max[2].item()
                    # )
                    # wind_force_radius = (self.obj_valid_max[0] - self.obj_valid_min[0]) * 10.
                    # wind_force_direction = (
                    #     self.obj_valid_max[0] - self.obj_valid_min[0],
                    #     self.obj_valid_max[1] - self.obj_valid_min[1],
                    #     self.obj_valid_min[2] - self.obj_valid_max[2]
                    # )
                    # force_field = gs.force_fields.Wind(
                    #     direction = wind_force_direction,
                    #     strength = 2,
                    #     radius = wind_force_radius,
                    #     center = wind_force_center,
                    # )
                    # force_field.activate()
                    # self.scene.add_force_field(
                    #     force_field = force_field
                    # )
                    pass
                
                elif self.config['example_name'] == "snow_man":
                    wind_force_center = (
                        self.obj_valid_min[0].item(),
                        self.obj_valid_min[1].item(),
                        self.obj_valid_max[2].item()
                    )
                    wind_force_radius = (self.obj_valid_max[0] - self.obj_valid_min[0]) * 10.
                    wind_force_direction = (
                        (self.obj_valid_max[0] - self.obj_valid_min[0]) * 0.2,
                        (self.obj_valid_max[1] - self.obj_valid_min[1]) * 0.,
                        (self.obj_valid_min[2] - self.obj_valid_max[2]) * 0.3
                    )
                    force_field = gs.force_fields.Wind(
                        direction = wind_force_direction,
                        strength = 4,
                        radius = wind_force_radius,
                        center = wind_force_center,
                    )
                    force_field.activate()
                    self.scene.add_force_field(
                        force_field = force_field
                    )

            elif obj_material_type == "mpm_sand":
                if self.config['example_name'] == "sand_castle":
                    wind_force_center = (
                        obj_max[0].item(),
                        obj_max[1].item(),
                        obj_max[2].item()
                    )
                    wind_force_radius = (obj_max[0] - obj_min[0]).item() * 10.
                    wind_force_direction = (
                        (obj_min[0] - obj_max[0]).item(),
                        (obj_min[1] - obj_max[1]).item(),
                        (obj_min[2] - obj_max[2]).item()
                    )
                    force_field = gs.force_fields.CustomWind(
                        direction = wind_force_direction,
                        strength = 5.0,
                        radius = wind_force_radius,
                        center = wind_force_center,
                    )
                    force_field.activate()
                    self.scene.add_force_field(
                        force_field = force_field
                    )
                
                elif self.config['example_name'] == "sand_house":
                    wind_force_center = (
                        (obj_min[0] + obj_max[0]).item() / 2,
                        (obj_min[1] + obj_max[1]).item() / 2,
                        (obj_min[2] + obj_max[2]).item() / 2,
                    )
                    wind_force_radius = (obj_max[0] - obj_min[0]).item() * 10.
                    wind_force_direction = (
                        (obj_min[0] - obj_max[0]).item(),
                        0,
                        (obj_min[2] - obj_max[2]).item()
                    )
                    force_field = gs.force_fields.CustomWind(
                        direction = wind_force_direction,
                        strength = 200.0,
                        radius = wind_force_radius,
                        center = wind_force_center,
                    )
                    force_field.activate()
                    self.scene.add_force_field(
                        force_field = force_field
                    )

            elif obj_material_type == "rigid":
                if self.config['example_name'] == "food_wine_1":
                    wind_force_center = (
                        obj_min[0].item(),
                        obj_max[1].item(),
                        obj_max[2].item()
                    )
                    wind_force_radius = (obj_max[0] - obj_min[0]).item() * 10.
                    wind_force_direction = (
                        (obj_max[0] - obj_min[0]).item(),
                        (obj_min[1] - obj_max[1]).item() * 2,
                        (obj_min[2] - obj_max[2]).item()
                    )
                    force_field = gs.force_fields.Wind(
                        direction = wind_force_direction,
                        strength = 1000.0,
                        radius = wind_force_radius,
                        center = wind_force_center,
                    )
                    force_field.activate()
                    # self.scene.add_force_field(
                    #     force_field = force_field
                    # )

            elif obj_material_type == "pbd_elastic":
                if self.config['example_name'] == "food_wine_2":
                    # emitter pos is the point location of object_1 with smallest x
                    cup_pts = self.objs[1].vmesh.verts
                    cup_pts = torch.tensor(cup_pts).to(self.device)
                    # Get x coordinates and find threshold at 30th percentile
                    x_coords = cup_pts[:, 0]
                    x_threshold = torch.quantile(x_coords, 0.15)
                    
                    # Filter points with x < threshold
                    filter_mask = x_coords < x_threshold
                    filter_pts = cup_pts[filter_mask]
                    
                    # Get z coordinates of filtered points and find threshold at 20th percentile
                    filter_pts_z = filter_pts[:, 2]
                    z_threshold = torch.quantile(filter_pts_z, 0.2)
                    
                    # Find point with z closest to threshold
                    z_dists = torch.abs(filter_pts_z - z_threshold)
                    min_z_idx = torch.argmin(z_dists)
                    min_x_idx = torch.where(filter_mask)[0][min_z_idx]
                    emitter_pos = cup_pts[min_x_idx].cpu().reshape(3).numpy()
                    self.emitter = self.scene.add_emitter(
                        material = gs.materials.MPM.Liquid(
                            E = self.config['MPM_E'],
                            nu = self.config['MPM_nu'],
                            rho = self.config['MPM_rho'],
                            viscous = self.config['MPM_viscous'],
                        ),
                        max_particles = 100000, # set a large enough number
                        surface = gs.surfaces.Rough(
                            color=(0.0, 0.9, 0.4, 1.0),
                        ),
                    )
                    self.emitter_pos = emitter_pos
                    self.emitter_dir = np.array([-0.2, 0, -1])
                    
                    cur_features_dc = self.obj_gaussians[1]["features_dc"]
                    emitter_features_dc = cur_features_dc[min_x_idx].reshape(1, 1, 3)
                    self.emitter_features_dc = emitter_features_dc

            elif obj_material_type == "pbd_cloth":

                # for cloth_hang_two
                if self.config['example_name'] == "cloth_hang_two":
                    wind_force_center = (
                        obj_min[0].item(),
                        (obj_min[1] + obj_max[1]).item() / 2,
                        (obj_min[2] + obj_max[2]).item() / 2,
                    )
                    wind_force_radius = (obj_max[0] - obj_min[0]).item() * 3.

                    if self.config['wind_direction_reverse']:
                        force_field = gs.force_fields.CustomWind(
                            direction = (1, 0, 0),
                            strength = 10.0,
                            radius = wind_force_radius,
                            center = wind_force_center,
                        )
                    else:
                        force_field = gs.force_fields.Constant(
                            direction = (1, 0, 0),
                            strength = 10.0,
                        )

                    force_field.activate()
                    self.scene.add_force_field(
                        force_field = force_field
                    )
                
                elif self.config['example_name'] == "kite_1":

                    wind_force_center = (
                        (obj_min[0] + obj_max[0]).item() / 2,
                        (obj_min[1] + obj_max[1]).item() / 2,
                        (obj_min[2] + obj_max[2]).item() / 2,
                    )
                    force_field = gs.force_fields.Vortex(
                        center = wind_force_center,
                        strength_perpendicular = 10.0,
                        strength_radial = 1.0,
                        falloff_pow = 2.0,
                        falloff_min = 0.01,
                        falloff_max = np.inf,
                    )
                    force_field.activate()
                    self.scene.add_force_field(
                        force_field = force_field
                    )
                
                elif self.config['example_name'] == "kite_2":
                    wind_force_center = (
                        (obj_min[0] + obj_max[0]).item() / 2,
                        (obj_min[1] + obj_max[1]).item() / 2,
                        (obj_min[2] + obj_max[2]).item() / 2 - (obj_max[2] - obj_min[2]).item() * 0.2,
                    )

                    if self.config['wind_direction_y']:
                        # genesis doesn't actually support direction in vortex force field
                        force_field = gs.force_fields.VortexCustom(
                            direction = (0, 1, 0),
                            center = wind_force_center,
                            strength_perpendicular = 100.0,
                            strength_radial = 50.0,
                            falloff_pow = 2.0,
                            falloff_min = 0.01,
                            falloff_max = np.inf,
                        )
                    else:
                        force_field = gs.force_fields.Vortex(
                            direction = (0, 0, 1),
                            center = wind_force_center,
                            strength_perpendicular = 300.0,
                            strength_radial = 10.0,
                            falloff_pow = 2.0,
                            falloff_min = 0.01,
                            falloff_max = np.inf,
                        )
                    force_field.activate()
                    self.scene.add_force_field(
                        force_field = force_field
                    )
            
            elif obj_material_type == "pbd_particle":
                if self.config['example_name'] == "demo_1" and idx == 0:
                    stick_pts = self.objs[1].vmesh.verts
                    stick_pts = torch.tensor(stick_pts).to(self.device)
                    z_coords = stick_pts[:, 2]
                    z_threshold_min = z_coords.min() + (z_coords.max() - z_coords.min()) * 0.2
                    z_threshold_max = z_coords.min() + (z_coords.max() - z_coords.min()) * 0.3
                    z_indices = torch.where((z_coords >= z_threshold_min) & (z_coords <= z_threshold_max))[0]
                    # Get points within z threshold range
                    filtered_pts = stick_pts[z_indices]
                    
                    # Find x target position (midpoint)
                    x_min, x_max = filtered_pts[:, 0].min(), filtered_pts[:, 0].max()
                    x_target = x_min + 0.5 * (x_max - x_min)
                    
                    # Find point closest to x target
                    x_dists = torch.abs(filtered_pts[:, 0] - x_target)
                    min_x_idx = torch.argmin(x_dists)
                    emitter_idx = z_indices[min_x_idx]
                    emitter_pos = stick_pts[emitter_idx].cpu().reshape(3).numpy()
                    self.emitter = self.scene.add_emitter(
                        material = gs.materials.MPM.Liquid(
                            E = self.config['MPM_E'],
                            nu = self.config['MPM_nu'],
                            rho = self.config['MPM_rho'],
                            viscous = self.config['MPM_viscous'],
                        ),
                        max_particles = 100000, # set a large enough number
                        surface = gs.surfaces.Rough(
                            color=(0.0, 0.9, 0.4, 1.0),
                        ),
                    )
                    self.emitter_pos = emitter_pos
                    self.emitter_dir = np.array([0, 0, -1])
                    cur_features_dc = self.obj_gaussians[1]["features_dc"]
                    emitter_features_dc = cur_features_dc[min_x_idx].reshape(1, 1, 3)
                    self.emitter_features_dc = emitter_features_dc

        self.scene.build()
        print("scene construction finished")
        if self.env is not None:
            # env is always the first entity in solver
            self.env.solver.fix_all_particle(0, env_particle_num)
            print("env particle fixed")
        
        if len(self.pbd_cloth_fix) > 0:
            for obj_id, points in self.pbd_cloth_fix.items():
                for point in points:
                    self.objs[obj_id].fix_particle(self.objs[obj_id].find_closest_particle(point))

    def set_object_linear_velocity(self, velocity, obj_id=0):
        """Set one rigid object's linear velocity without angular velocity."""
        if self.len_obj != 1:
            raise ValueError("Interaction V1 supports exactly one object")
        if obj_id != 0:
            raise ValueError("Interaction V1 only supports obj_id=0")
        if self.material_types[obj_id] != "rigid":
            raise NotImplementedError(
                "Interaction V1 velocity transfer currently supports rigid objects only"
            )

        velocity = torch.as_tensor(
            velocity, dtype=torch.float32, device=self.device
        ).reshape(-1)
        if velocity.numel() != 3 or not torch.isfinite(velocity).all():
            raise ValueError("Object linear velocity must contain three finite values")

        qvel = torch.zeros(
            self.objs[obj_id].n_dofs, dtype=torch.float32, device=self.device
        )
        if qvel.numel() != 6:
            raise RuntimeError(
                f"Expected a free rigid object with 6 DoFs, got {qvel.numel()}"
            )
        qvel[:3] = velocity
        self.objs[obj_id].set_dofs_velocity(qvel)
        print(
            "[interaction] Set Genesis object linear velocity:",
            velocity.detach().cpu().numpy(),
        )
        
    def env_process(self, env_xyz, obj_valid_min, obj_valid_max, obj_min, obj_max):
        # by option keep only the environment within the object range
        # x_mask = (env_xyz[:, 0] >= obj_valid_min[0]) & (env_xyz[:, 0] <= obj_valid_max[0])
        # y_mask = (env_xyz[:, 1] >= obj_valid_min[1]) & (env_xyz[:, 1] <= obj_valid_max[1])

        x_mask = (env_xyz[:, 0] >= obj_min[0]) & (env_xyz[:, 0] <= obj_max[0])
        y_mask = (env_xyz[:, 1] >= obj_min[1]) & (env_xyz[:, 1] <= obj_max[1])
        z_mask = env_xyz[:, 2] <= obj_min[2]

        # Combine masks to get points within the specified region
        region_mask = x_mask & y_mask & z_mask

        # Select points within the region
        selected_pts = env_xyz[region_mask]

        if 'hack_env' in self.config and self.config['hack_env']:
            hack = True
        else:
            hack = False

        if selected_pts.shape[0] == 0 or hack:
            print("hack setting one environment point as plane")
            # Create uniform grid points in x and y
            x_num = int((obj_valid_max[0] - obj_valid_min[0]) / 0.01)
            y_num = int((obj_valid_max[1] - obj_valid_min[1]) / 0.01)
            x = torch.linspace(obj_valid_min[0], obj_valid_max[0], x_num).to(self.device)
            y = torch.linspace(obj_valid_min[1], obj_valid_max[1], y_num).to(self.device)
            
            # Create meshgrid and reshape to [N,2]
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            xy_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
            
            # Add z coordinate
            # z = torch.full((xy_points.shape[0], 1), obj_valid_min[2] - 0.2 * (obj_min[2] - obj_valid_min[2]), device=self.device)
            z = torch.full((xy_points.shape[0], 1), obj_min[2], device=self.device)
            selected_pts = torch.cat([xy_points, z], dim=1)

        return selected_pts


    def simulate_step(self, sid, simulation_steps, save_dir, gaussian_vis_freq):
        if sid == 0:
            self.cam.start_recording()
            render_out = self.cam.render()
            cv2.imwrite(os.path.join(save_dir, f"render_rgb_{sid:04d}_begin.png"), render_out[0])
        
        if self.force_function is not None:
            for obj_id in range(self.len_obj):
                obj_material_type = self.material_types[obj_id]
                if obj_material_type == 'rigid':
                    _ = self.force_function(self.objs[obj_id], obj_id, sid, self.objs)
                
                else:
                    pass
        
        if self.emitter is not None:
            pass

        self.scene.step()
        render_out = self.cam.render()

        sim_out = dict()

        if sid % gaussian_vis_freq == 0:
            for obj_id in range(self.len_obj):
                gaussian_start_pos = self.obj_gaussians[obj_id]["xyz"]
                num_vertices = self.num_vertices[obj_id]
                # verts_particles_indices = self.verts_closest_indices[obj_id]
                # particles_start_pos_in_gs = self.particles_pos_start_in_gs[obj_id]

                obj_material_type = self.material_types[obj_id]

                # if obj_id in self.config['rigid_id_list']:
                #     particles_now_pos_in_gs = self.objs[obj_id].get_verts()
                #     particles_now_pos_in_gs = particles_now_pos_in_gs.cpu().numpy()

                #     particles_change_pos_in_gs = torch.tensor(particles_now_pos_in_gs - particles_start_pos_in_gs).to(self.device)
                
                #     verts_change_pos_in_gs = particles_change_pos_in_gs[verts_particles_indices]
                #     verts_change_pos_in_pt3d = gs_to_pt3d(verts_change_pos_in_gs)
                #     gaussian_now_pos = gaussian_start_pos + verts_change_pos_in_pt3d
                
                if obj_material_type == "pbd_gas":
                    # the genesis pbd currently doesn't support remesh a gas for it in a easy way
                    # so here directly use nearest neighbor to get original mesh vertices positions updates
                    verts_particles_indices = self.verts_closest_indices[obj_id]

                    particles_now_pos_in_gs = self.objs[obj_id].solver.particles.pos.to_numpy()[self.objs[obj_id].particle_start:self.objs[obj_id].particle_end]
                    particles_start_pos_in_gs = self.objs[obj_id].init_particles

                    particles_now_pos_in_gs = torch.tensor(particles_now_pos_in_gs).to(self.device)
                    particles_start_pos_in_gs = torch.tensor(particles_start_pos_in_gs).to(self.device)

                    particles_change_pos_in_gs = particles_now_pos_in_gs - particles_start_pos_in_gs
                    verts_change_pos_in_gs = particles_change_pos_in_gs[verts_particles_indices]
                    # average the change pos of the num_closest nearest neighbors
                    verts_change_pos_in_gs = verts_change_pos_in_gs.mean(dim=1)
                    verts_change_pos_in_pt3d = gs_to_pt3d(verts_change_pos_in_gs)
                    gaussian_now_pos = gaussian_start_pos + verts_change_pos_in_pt3d
                
                elif obj_material_type.startswith("pbd") or (obj_material_type in ["mpm_elastic", "mpm_elastoplastic"]):
                    vertices_now_pos_in_gs = self.objs[obj_id].solver.vverts_render.pos.to_numpy()
                    v_start = self.objs[obj_id].vvert_start
                    n_vertices = self.objs[obj_id].n_vverts
                    vertices_now_pos_in_gs = vertices_now_pos_in_gs[v_start:v_start+n_vertices]
                    vertices_now_pos_in_gs = torch.tensor(vertices_now_pos_in_gs).to(self.device)
                    vertices_now_pos_in_pt3d = gs_to_pt3d(vertices_now_pos_in_gs)
                    gaussian_now_pos = vertices_now_pos_in_pt3d
                
                elif obj_material_type == "rigid":
                    if self.rigid_mesh_vertices[obj_id] is None:
                        vertices_mesh = self.objs[obj_id].vgeoms[0].get_trimesh()
                        vertices_numpy = np.array(vertices_mesh.vertices)
                        vertices = torch.tensor(vertices_numpy).to(self.device).float()
                        self.rigid_mesh_vertices[obj_id] = vertices
                    else:
                        vertices = self.rigid_mesh_vertices[obj_id]
                    
                    rigid_T = self.objs[obj_id].solver._vgeoms_render_T
                    rigid_idx = self.objs[obj_id].idx
                    rigid_T = torch.tensor(rigid_T[rigid_idx,0]).to(self.device).float()
                    vertices_h = torch.cat([vertices, torch.ones(vertices.shape[0], 1).to(self.device)], dim=1)
                    vertices_now_pos_in_gs = torch.matmul(rigid_T.unsqueeze(0), vertices_h.unsqueeze(-1)).squeeze(-1)[:, :3]
                    vertices_now_pos_in_pt3d = gs_to_pt3d(vertices_now_pos_in_gs)
                    gaussian_now_pos = vertices_now_pos_in_pt3d
                
                elif obj_material_type.startswith("fem"):
                    vertices_now_pos_in_gs, _ = self.objs[obj_id].solver.get_state_render(self.objs[obj_id].sim.cur_substep_local)
                    vertices_now_pos_in_gs = vertices_now_pos_in_gs.to_numpy(dtype="float")
                    v_start = self.objs[obj_id].v_start
                    n_vertices = self.objs[obj_id].n_vertices
                    vertices_now_pos_in_gs = vertices_now_pos_in_gs[v_start:v_start+n_vertices]
                    vertices_now_pos_in_gs = torch.tensor(vertices_now_pos_in_gs).to(self.device)
                    vertices_now_pos_in_pt3d = gs_to_pt3d(vertices_now_pos_in_gs)
                    gaussian_now_pos = vertices_now_pos_in_pt3d

                elif obj_material_type in [
                    "mpm_sand",
                    "mpm_liquid",
                    "mpm_snow"
                ]:
                    # granular object doesn't maintain the mesh structure in genesis
                    # changed genesis code to support skinning in granular material type
                    particles_now_pos_in_gs = self.objs[obj_id].solver.vverts_render.pos.to_numpy()
                    v_start = self.objs[obj_id].vvert_start
                    n_vertices = self.objs[obj_id].n_vverts
                    particles_now_pos_in_gs = particles_now_pos_in_gs[v_start:v_start+n_vertices]
                    particles_now_pos_in_gs = torch.tensor(particles_now_pos_in_gs).to(self.device)
                    particles_now_pos_in_pt3d = gs_to_pt3d(particles_now_pos_in_gs)
                    gaussian_now_pos = particles_now_pos_in_pt3d

                else:
                    raise ValueError(f"Object {obj_id} with material type {obj_material_type} is not supported")

                sim_out.update({
                    f"obj_{obj_id:04d}": {
                        "xyz": gaussian_now_pos.float(),
                        "rotation": self.obj_gaussians[obj_id]["rotation"],
                        "scaling": self.obj_gaussians[obj_id]["scaling"],
                        "features_dc": self.obj_gaussians[obj_id]["features_dc"],
                        "opacity": self.obj_gaussians[obj_id]["opacity"],
                        "valid_mask": torch.ones(num_vertices, 1).to(self.device),
                    }
                })
            
            # when there is emitter, everytime we get all emitted particles
            # now only assume there is one emitter
            if self.emitter is not None:
                pass

            cv2.imwrite(os.path.join(save_dir, f"render_rgb_{sid:04d}.png"), render_out[0])
        
        if sid == simulation_steps - 1:
            self.cam.stop_recording(save_to_filename=os.path.join(save_dir, "render_rgb.mp4"), fps=20)
        
        return sim_out
