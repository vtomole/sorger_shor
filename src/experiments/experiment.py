from shor import run_shor
from time_tracker import TimeTracker

from e1.script import run as run_e1
from e2.script import run as run_e2
from e1.script import run as run_e3
from e4.script import run as run_e4


def run_experiment_1():
    """
    Run experiment 1
    """
    run_e1(run_shor, TimeTracker)
    return

def run_experiment_2():
    """
    Run experiment 2
    """
    run_e2(TimeTracker)

def run_experiment_3():
    """
    Run experiment 3
    """
    run_e3(run_shor, TimeTracker)
    return

def run_experiment_4():
    """
    Run experiment 4
    """
    run_e4(run_shor)
    return


if __name__ == "__main__":
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    run_experiment_4()
