from simulators.inverse_kinematics_simulator import InverseKinematicsSimulator
from simulators.two_moons_simulator import TwoMoonsSimulator
from simulators.slcp_simulator import SLCPSimulator


def get_simulator(simulator_name):
    if simulator_name == 'SLCP':
        return SLCPSimulator()
    elif simulator_name == '2dMoon':
        return TwoMoonsSimulator()
    elif simulator_name == 'IK':
        return InverseKinematicsSimulator()
    else:
        raise ValueError('Invalid simulator name. Valid names are: "SLCP", "2dMoon", "IK".')
