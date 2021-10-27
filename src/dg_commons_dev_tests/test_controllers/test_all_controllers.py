from dg_commons_dev_tests.test_controllers.controllers_to_test import *
from dg_commons_dev_tests.test_controllers.controller_test_utils import Select, Verbosity, \
    TestMultipleControllerInstances
from dg_commons_dev.analysis.metrics import *
from dg_commons_dev_tests.test_controllers.controller_scenarios.scenario_to_test import scenarios

if __name__ == '__main__':
    verbosity: Verbosity = Verbosity(2)

    scenarios_to_test = [Select(scenarios["lane_change_left"], False),
                         Select(scenarios["turn_90_right"], True),
                         Select(scenarios["turn_90_left"], False),
                         Select(scenarios["small_snake"], False),
                         Select(scenarios["u-turn"], False),
                         Select(scenarios["left_cont_curve"], False),
                         Select(scenarios["vertical"], False),
                         Select(scenarios["emergency_brake"], False),
                         Select(scenarios["cruise_control"], False),
                         Select(scenarios["race"], False)]

    controllers_to_test = [
        Select(TestLQR, False),
        Select(TestPurePursuit, True),
        Select(TestStanley, False),
        Select(TestNMPCFullKinContPV, True), Select(TestNMPCFullKinContAN, False),
        Select(TestNMPCFullKinDisPV, False), Select(TestNMPCFullKinDisAN, False),
        Select(TestNMPCLatKinContPV, True), Select(TestNMPCLatKinContAN, False),
        Select(TestNMPCLatKinDisPV, False), Select(TestNMPCLatKinDisAN, False)
    ]

    metrics_to_test = [
                    Select(DeviationLateral, True),
                    Select(DeviationVelocity, True),
                    Select(SteeringVelocity, True),
                    Select(Acceleration, True),
                    Select(DTForCommand, True)
    ]

    Test = TestMultipleControllerInstances(controllers_to_test, metrics_to_test, scenarios_to_test, verbosity)
    Test.run()
