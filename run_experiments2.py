import run_experiments1
import sys

test_values = [
    0.01, 0.1, 0.2, 0.5
]

def main(num_proc):

    pt2_tasks = []

    for x in test_values:
        pt2_tasks.append([False, 0.05, x, "Beta: {}".format(x)])

    pt2_tasks = [ ("Beta Values", pt2_tasks) ]

    total_experiments = 0
    for task in run_experiments1.pt1_tasks:
        total_experiments += len(task[1])

    run_experiments1.main(num_proc, pt2_tasks, total_experiments, len(run_experiments1.pt1_tasks))

if __name__ == "__main__":

    if len(sys.argv) <= 1:
        num_proc = 1
    else:
        num_proc = sys.argv[1]

    if num_proc is None:
        num_proc = 1
    else:
        num_proc = int(num_proc)

    main(num_proc)
    