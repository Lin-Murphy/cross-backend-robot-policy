"""Gym ALOHA adapter: retain native observations, rewards and termination."""
import numpy as np
from .execution_contract import BackendCapabilities, BackendObservation, ActionReceipt, StopReceipt


class AlohaBackendAdapter:
    def __init__(self, env, observation, episode):
        self.env, self.observation, self.episode = env, observation, episode
        self.index = 0
        self.stopped = False
        names = tuple(f'{side}_{joint}' for side in ('left', 'right')
                      for joint in ('waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate', 'gripper'))
        self.capabilities = BackendCapabilities('aloha', names,
            (('rad',) * 6 + ('normalized_0_1',)) * 2,
            tuple(observation['pixels']), 'simulation', 'automatic', 'sim_truth', 'simulation_step', False)

    def observe(self):
        identity = f'aloha:{self.episode}:{self.index}'
        return BackendObservation(identity, self.capabilities.joint_names,
            self.capabilities.action_units, tuple(float(x) for x in self.observation['agent_pos']),
            self.observation['pixels'], {k: identity + ':' + k for k in self.capabilities.cameras},
            {k: None for k in self.capabilities.cameras}, None, 'simulation')

    def dispatch(self, request):
        if self.stopped:
            return ActionReceipt(False, request.target, 'simulation_step', 0, None, 'backend_stopped')
        self.transition = self.env.step(np.asarray(request.target))
        self.observation = self.transition[0]
        self.index += 1
        return ActionReceipt(True, request.target, 'simulation_step', 0, None)

    def stop(self):
        self.stopped = True
        return StopReceipt(True, 'gym_episode_latched', 0, 'simulation_step')
