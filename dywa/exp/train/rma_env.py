#!/usr/bin/env python3

import copy
from typing import Optional, Tuple, List
from dataclasses import dataclass

from env.env.wrap.normalize_env import NormalizeEnv
from util.math_util import matrix_to_pose9d, pose7d_to_matrix, pose9d_to_matrix
import torch as th
import einops
from gym import spaces

from env.env.wrap.base import (add_obs_field,
                                   WrapperEnv,
                                   ObservationWrapper)
from env.env.wrap.popdict import PopDict
from models.common import (
    map_tensor
)
from models.cloud.point_mae import (
    subsample
)

from pytorch3d.ops import sample_farthest_points

class CombineCloud(ObservationWrapper):
    '''
    Combine multiple partial clouds into one
    called: init, partial = 0, reset/step
    '''
    @dataclass
    class Config:
        src_keys: Tuple[str, ...] = (
            'partial_cloud',
            'partial_cloud_1',
            'partial_cloud_2')
        dst_key: str = 'partial_cloud'
        cloud_size: int = 512
        combine_method: str = "uniform"

    def __init__(self, cfg, env):
        super().__init__(env, self._wrap_obs)
        self.cfg = cfg
        obs_space = env.observation_space
        cloud_space = spaces.Box(-float('inf'),
                                 +float('inf'),
                                 shape=(cfg.cloud_size, 3))
        self._obs_space, self._update_fn = add_obs_field(
            obs_space,
            cfg.dst_key,
            cloud_space)

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        cfg = self.cfg
        # combine
        cloud = th.cat([obs[k] for k in cfg.src_keys],
                       dim=-2)
        # reduce
        if self.cfg.combine_method == "uniform":
            cloud = subsample(cloud, n=cfg.cloud_size) ## FPS???
        elif self.cfg.combine_method == "FPS":
            cloud = sample_farthest_points(cloud, K = cfg.cloud_size)[0]
        else:
            raise NotImplementedError
        
        # update
        obs = self._update_fn(obs, cloud)
        return obs


class PerturbCloud(ObservationWrapper):
    '''
    called: init-partial=0/rel, reset/step, 

    '''
    @dataclass
    class Config:
        noise_mag: float = 0.005
        noise_type: str = 'additive'
        key: str = 'partial_cloud'

    def __init__(self, cfg: Config, env):
        super().__init__(env, self._wrap_obs)
        self.cfg = cfg
        self._obs_space = env.observation_space

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        cfg = self.cfg

        # Make a shallow copy
        obs = dict(obs)

        cloud = obs.pop(cfg.key)

        # sample noise
        noise = cfg.noise_mag * th.randn(
            (*cloud.shape[:-1], 3),
            dtype=cloud.dtype, device=cloud.device
        )

        # add noise
        # random gaussian noise
        if self.cfg.noise_type == 'additive':
            cloud = cloud + noise
        # noise proportional to distance
        elif self.cfg.noise_type == 'scaling':
            cloud = cloud * (1 + noise)
        else:
            raise ValueError(
                f"{cfg.noise_type} is not a proper noise type")

        obs[cfg.key] = cloud
        return obs


class PerturbGoal(ObservationWrapper):
    '''
    called: init-partial=0/rel, reset/step
    '''
    @dataclass
    class Config:
        noise_mag: float = 0.005
        key: str = 'goal'

    def __init__(self, cfg: Config, env):
        super().__init__(env, self._wrap_obs)
        self.cfg = cfg
        self._obs_space = env.observation_space

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        cfg = self.cfg

        # Make a shallow copy
        obs = dict(obs)
        goal = obs.pop(cfg.key)
        goal = goal + (cfg.noise_mag * th.randn_like(goal))
        obs[cfg.key] = goal
        return obs


class AddTeacherState(WrapperEnv):
    def __init__(self,
                 env,
                 teacher,
                 state_size: int,
                 key: str = 'teacher_state',
                 ):
        super().__init__(env)
        self.__key = key
        self.teacher = teacher
        state_space = spaces.Box(-float('inf'), +float('inf'),
                                 shape=(state_size,))
        self._obs_space, self._update_fn = add_obs_field(
            env.observation_space,
            key,
            state_space
        )
        self.__memory = None
        self.teacher.eval()

    @property
    def observation_space(self):
        return self._obs_space

    def reset(self):
        obs = super().reset()
        _, state, self.__memory, done = self.teacher.init(obs)
        obs = self._update_fn(obs, state)
        return obs

    def step(self, actions: th.Tensor):
        obs, rew, done, info = super().step(actions)
        with th.inference_mode():
            state, self.__memory = self.teacher.state_net(
                self.__memory,
                actions, obs)
        obs = self._update_fn(obs, state.clone())

        # Reset `memory` where done=True.
        # we reset memory after state.clone()
        # just in case it aliases the underlying
        # memory buffer.
        with th.inference_mode():
            keep = (~done)[..., None]
            map_tensor(self.__memory,
                       lambda src, _: src.mul_(keep))
        return obs, rew, done, info


class AddTeacherAction(WrapperEnv):
    def __init__(self,
                 env,
                 teacher,
                 key: str = 'teacher_action',
                 ):
        super().__init__(env)
        self.__key = key
        self.teacher = teacher

        # mu+ls
        num_act: int = env.action_space.shape[0]
        action_spaces = spaces.Box(-float('inf'), +float('inf'),
                                   shape=(2, num_act,))
        self._obs_space, self._update_fn = add_obs_field(
            env.observation_space,
            key,
            action_spaces
        )
        self.__memory = None
        self.teacher.eval()

    @property
    def observation_space(self):
        return self._obs_space

    def reset(self):
        obs = super().reset()
        _, state, self.__memory, done = self.teacher.init(obs)

        mu, ls = self.teacher.actor_net(state)
        ls = einops.repeat(ls, '... -> n ...',
                           n=mu.shape[0])
        muls = th.stack([mu, ls], dim=-2)
        obs = self._update_fn(obs, muls)
        return obs

    def step(self, actions: th.Tensor):
        obs, rew, done, info = super().step(actions)
        with th.inference_mode():
            state, self.__memory = self.teacher.state_net(
                self.__memory,
                actions, obs)
            mu, ls = self.teacher.actor_net(state)
            ls = einops.repeat(ls, '... -> n ...',
                               n=mu.shape[0])
            # muls = th.cat([mu, ls], dim=-1)
            muls = th.stack([mu, ls], dim=-2)

        obs = self._update_fn(
                obs, muls.clone())

        # Reset `memory` where done=True.
        # we reset memory after state.clone()
        # just in case it aliases the underlying
        # memory buffer.
        with th.inference_mode():
            keep = (~done)[..., None]
            map_tensor(self.__memory,
                       lambda src, _: src.mul_(keep))
        return obs, rew, done, info


class AddStudentState(WrapperEnv):
    def __init__(self, env, student,
                 state_size: int,
                 key: str = 'student_state'):
        super().__init__(env)
        self.__key = key
        self.student = student
        state_space = spaces.Box(-float('inf'), +float('inf'),
                                 shape=(state_size,))
        self._obs_space, self._update_fn = add_obs_field(
            env.observation_space,
            key,
            state_space)
        self.__memory = None
        self.student.eval()

    @property
    def observation_space(self):
        return self._obs_space

    def reset(self):
        obs = super().reset()
        state = self.student.reset(obs)

        # also update internal data
        done = th.zeros((self.num_env,),
                        dtype=bool,
                        device=self.device)
        self.student.reset_state(done)

        obs = self._update_fn(obs, state)
        return obs

    def step(self, actions: th.Tensor):
        obs, rew, done, info = super().step(actions)

        # Run inference and add state to `obs`.
        with th.inference_mode():
            state = self.student(obs, 0, done)
        obs = self._update_fn(obs, state.clone())

        # Reset `memory` where done=True.
        # we reset memory after state.clone()
        # just in case it aliases the underlying
        # memory buffer.
        with th.inference_mode():
            self.student.reset_state(done)

        return obs, rew, done, info
       
class AddRelGoalCloud(ObservationWrapper):
    '''
    called: rel, reset/step
    '''
    
    @dataclass
    class Config:
        src_keys: str = 'partial_cloud'
        dst_key: str = 'goal_cloud'
        cloud_size: int = 512

    def __init__(self, cfg, env):
        super().__init__(env, self._wrap_obs)
        self.cfg = cfg
        obs_space = env.observation_space
        cloud_space = spaces.Box(-float('inf'),
                                +float('inf'),
                                shape=(cfg.cloud_size, 3))
        self._obs_space, self._update_fn = add_obs_field(
            obs_space,
            cfg.dst_key,
            cloud_space)

        self.used_keys = ['partial_cloud', 'abs_goal', 'object_state']

        normalizer = self.unwrap(target=NormalizeEnv).normalizer
        normalizer.obs_rms[self.cfg.dst_key] = copy.deepcopy(normalizer.obs_rms[self.cfg.src_keys])

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        norm = self.unwrap(target=NormalizeEnv)
        un_obs = {key: obs[key] for key in obs if key in self.used_keys}
        un_obs = norm.normalizer.unnormalize_obs(un_obs)
        
        object_pose = pose7d_to_matrix(un_obs['object_state'][:, :7])
        goal_pose = pose9d_to_matrix(un_obs['abs_goal'])
        rel_goal = th.bmm(goal_pose, th.linalg.inv(object_pose))
        goal_pc = th.bmm(un_obs['partial_cloud'], rel_goal[:, :3, :3].transpose(1, 2)) + rel_goal[:, :3, 3].unsqueeze(1)  

        t_obs = {}
        t_obs['partial_cloud'] = goal_pc.clone()
        goal_pc = norm.normalizer.normalize_obs(t_obs)['partial_cloud']

        obs = self._update_fn(obs, goal_pc)
        return obs

def _get_scene_from_env(env) -> Optional[object]:
    """Unwrap until we get an env that has .scene (e.g. PushEnv)."""
    e = env
    while e is not None:
        scene = getattr(e, 'scene', None)
        if scene is not None:
            return scene
        e = getattr(e, 'env', None)
    return None


def _upsample_cloud(pc: th.Tensor, target_n: int) -> th.Tensor:
    """Repeat points to get [..., target_n, 3]. pc shape [..., N, 3], N <= target_n."""
    n = pc.shape[-2]
    if n >= target_n:
        return pc[..., :target_n, :]
    # repeat indices to fill target_n
    idx = th.arange(target_n, device=pc.device) % n
    return pc[..., idx, :]


class AddInitialCloud(ObservationWrapper):
    """
    Record Initial Point Cloud.
    When scene provides cur_initial_cloud (e.g. PhyBlock canonical clouds),
    use it; otherwise use partial_cloud (upsampled to num_point if needed).
    """

    def __init__(self, env,
                 key: str = 'initial_cloud',
                 key_pc: str = 'partial_cloud',
                 num_point: int = 2048,
                 ):
        super().__init__(env, self._wrap_obs)
        self.num_point = num_point
        self.initial_cloud = th.zeros((env.num_env, num_point, 3),
                                      dtype=th.float,
                                      device=env.device)
        self.reset_indices = None

        obs_space, self.update_fn = add_obs_field(
            env.observation_space, key,
            spaces.Box(-float('inf'), +float('inf'), (num_point, 3))
        )
        self._obs_space = obs_space

        self.key = key
        self.key_pc = key_pc

        normalizer = self.unwrap(target=NormalizeEnv).normalizer
        # initial_cloud may have different n_point than key_pc; reuse same per-point stats
        normalizer.obs_rms[self.key] = copy.deepcopy(normalizer.obs_rms[self.key_pc])

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        scene = _get_scene_from_env(self.env)
        from_scene = (
            scene is not None
            and getattr(scene, 'cur_initial_cloud', None) is not None
        )
        if from_scene:
            src = scene.cur_initial_cloud
            if src.shape[-2] != self.num_point:
                src = _upsample_cloud(src, self.num_point)
        else:
            src = _upsample_cloud(obs[self.key_pc], self.num_point)

        if self.reset_indices is not None and self.reset_indices.numel() != 0:
            self.initial_cloud[self.reset_indices] = src[self.reset_indices]
        else:
            self.initial_cloud = src.clone()

        self.reset_indices = th.tensor([], device=self.env.device)
        obs = self.update_fn(obs, self.initial_cloud)
        return obs
    
    def reset_indexed(self, indices: Optional[th.Tensor] = None):

        out = super().reset_indexed(indices)
        if indices is not None:
            self.reset_indices = indices.clone()
        else:
            self.reset_indices = None

        return out
    
    def step(self, actions: th.Tensor):
        done = self.buffers['done']
        done_indices = th.argwhere(done).ravel()

        if done_indices is not None:
            self.reset_indices = done_indices.clone()
        else:
            self.reset_indices = None

        return super().step(actions)
    
class AddInitialRelGoal(ObservationWrapper):
    """
    Record Initial Point Cloud
    call: initial-partial=0, reset
    """

    def __init__(self, env,
                 key: str = 'initial_rel_goal',
                 key_goal: str = 'abs_goal',
                 key_pose: str = 'object_state',
                 use_6d: bool = False,
                 ):
        super().__init__(env, self._wrap_obs)

        self.__use_6d = use_6d

        n: int = 9 if use_6d else 7
        
        self.initial_rel_goal = th.zeros((env.num_env, n),
                                  dtype=th.float,
                                  device=env.device)
        self.reset_indices = None

        obs_space, self.update_fn = add_obs_field(
            env.observation_space,
            key, spaces.Box(-1.0, 1.0, (n,))
        )
        self._obs_space = obs_space

        self.key = key
        self.key_pose = key_pose
        self.key_goal = key_goal

        self.used_keys = [key_goal, key_pose]

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        norm = self.unwrap(target=NormalizeEnv)
        un_obs = {key: obs[key] for key in obs if key in self.used_keys}
        un_obs = norm.normalizer.unnormalize_obs(un_obs)

        if self.reset_indices is not None:
            if self.reset_indices.numel() != 0:
                self.initial_rel_goal[self.reset_indices] = matrix_to_pose9d(th.bmm(pose9d_to_matrix(un_obs[self.key_goal][self.reset_indices]),
                                th.linalg.inv(pose7d_to_matrix(un_obs[self.key_pose][self.reset_indices, :7]))))
        else:
            self.initial_rel_goal = matrix_to_pose9d(th.bmm(pose9d_to_matrix(un_obs[self.key_goal]),
                                th.linalg.inv(pose7d_to_matrix(un_obs[self.key_pose][:, :7]))))
        self.reset_indices = th.tensor([],device= self.env.device)
        obs = self.update_fn(obs, self.initial_rel_goal)
        return obs
    
    def reset_indexed(self, indices: Optional[th.Tensor] = None):
        out = super().reset_indexed(indices)
        if indices is not None:
            self.reset_indices = indices.clone()
        else:
            self.reset_indices = None

        return out
    
    def step(self, actions: th.Tensor):
        done = self.buffers['done']
        done_indices = th.argwhere(done).ravel()

        if done_indices is not None:
            self.reset_indices = done_indices.clone()
        else:
            self.reset_indices = None

        return super().step(actions)
    
class AddInitialGoalCloud(ObservationWrapper):
    '''
    用 initial_cloud + initial_rel_goal 算出 goal_cloud。
    若 obs 中已有 initial_cloud（如任务 2 由 scene 提供），则安全使用；
    否则回退到 partial_cloud（上采样到 cloud_size），避免 KeyError。
    '''
    @dataclass
    class Config:
        src_keys: str = 'initial_cloud'
        dst_key: str = 'goal_cloud'
        cloud_size: int = 2048

    def __init__(self, cfg, env):
        super().__init__(env, self._wrap_obs)
        self.cfg = cfg
        obs_space = env.observation_space
        cloud_space = spaces.Box(-float('inf'),
                                +float('inf'),
                                shape=(cfg.cloud_size, 3))
        self._obs_space, self._update_fn = add_obs_field(
            obs_space,
            cfg.dst_key,
            cloud_space)

        self.used_keys = ['initial_cloud', 'initial_rel_goal']

        normalizer = self.unwrap(target=NormalizeEnv).normalizer
        normalizer.obs_rms[self.cfg.dst_key] = copy.deepcopy(normalizer.obs_rms[self.cfg.src_keys])

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        norm = self.unwrap(target=NormalizeEnv)
        # 保护：优先使用 obs 里已有的 initial_cloud（任务 2 由 scene 提供），否则用 partial_cloud 回退
        if 'initial_cloud' in obs:
            base_pc = obs['initial_cloud']
            unnorm_key = self.cfg.src_keys
        else:
            base_pc = _upsample_cloud(obs['partial_cloud'], self.cfg.cloud_size)
            unnorm_key = 'partial_cloud'

        un_obs = {unnorm_key: base_pc}
        un_obs = norm.normalizer.unnormalize_obs(un_obs)
        initial_pc = un_obs[unnorm_key]

        initial_rel_goal = pose9d_to_matrix(obs['initial_rel_goal'])
        goal_pc = th.bmm(initial_pc, initial_rel_goal[:, :3, :3].transpose(1, 2)) + initial_rel_goal[:, :3, 3].unsqueeze(1)

        t_obs = {self.cfg.dst_key: goal_pc.clone()}
        goal_pc = norm.normalizer.normalize_obs(t_obs)[self.cfg.dst_key]

        obs = self._update_fn(obs, goal_pc)
        return obs


class ShuffleCloud(ObservationWrapper):
    @dataclass
    class Config:
        keys: Tuple[str, ...] = (
            'partial_cloud',
            'goal_cloud')

    def __init__(self, cfg: Config, env):
        super().__init__(env, self._wrap_obs)
        self.cfg = cfg
        self._obs_space = env.observation_space

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        cfg = self.cfg

        # Loop over each key and shuffle the corresponding point cloud
        for key in cfg.keys:
            if key in obs.keys():
                obs[key] = self._shuffle_cloud(obs[key])
            else:
                raise KeyError(f"Key '{key}' not found in observations.")

        return obs

    def _shuffle_cloud(self, cloud):
        """Shuffle the point cloud along the N dimension for each batch independently."""
        batch_size = cloud.shape[0]
        num_points = cloud.shape[1]
        indices = th.randperm(num_points).to(cloud.device)
        shuffled_cloud = cloud[:, indices]
        return shuffled_cloud
        
def setup_rma_env_v2(cfg, env, agent,
                     state_size: int,
                     is_student: bool = False,
                     dagger: bool = False):
    if not is_student:
        if dagger:
            env = AddTeacherAction(env, agent)
        if cfg.add_teacher_state:
            env = AddTeacherState(env, agent, state_size)
    if not cfg.use_partial_cloud:
        env = CombineCloud(cfg.combine_cloud, env)
    env = PerturbCloud(cfg.perturb_cloud, env)
    env = PerturbGoal(cfg.perturb_goal, env)
    env = PopDict(env, ['icp_emb'])
    # 先加 initial_cloud / goal_cloud，再加 AddStudentState，这样 student.reset(obs) 收到的 obs 里才有 goal_cloud
    if cfg.use_goal_cloud:
        if cfg.goal_cloud_type == 'rel':
            env = AddRelGoalCloud(cfg.rel_goal_cloud, env)
        elif cfg.goal_cloud_type == 'initial':
            env = AddInitialCloud(env)
            env = AddInitialRelGoal(env, use_6d= cfg.use_6d_rel_goal)
            env = AddInitialGoalCloud(cfg.initial_goal_cloud, env)
        else:
            raise NotImplementedError(cfg.goal_cloud_type)
    if is_student:
        env = AddStudentState(env, agent, state_size)
    if cfg.use_shuffle_cloud:
        env = ShuffleCloud(cfg.shuffle_cloud, env)
    return env
