from multiprocessing import Pool, Manager
import subprocess
import sys, os
import traceback
import csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
figure(num=None, figsize=(9, 6), dpi=120, facecolor='w', edgecolor='k')

NUM_RUNS = 3
NUM_ITERATIONS = 2000

pt1_tasks = [  
        ( "Alpha Tests",
        [
            [True, 0.01, 0.1, "Alpha: 0.01"],  # Sub task 1
            [True, 0.05, 0.1, "Alpha: 0.05"],
            [True, 0.1, 0.1, "Alpha: 0.1"],
            [True, 0.2, 0.1, "Alpha: 0.2"],
            [True, 0.4, 0.1, "Alpha: 0.4"]
        ]),
        ( "Epsilon Tests",
        [
            [False, 0.05, 0.01, "Epsilon: 0.01"], # Sub task 2
            [False, 0.05, 0.1, "Epsilon: 0.1"],
            [False, 0.05, 0.2, "Epsilon: 0.2"],
            [False, 0.05, 0.5, "Epsilon: 0.5"]
        ])
    ]

def generate_graph(input):

    print("Starting experiment {}, run {}".format(input[0], input[1]))

    try:

        experiment = input[2]
        output_directory = run_experiment(input[0], input[1], experiment[1], experiment[2])

        input[3][(input[0], input[1])] = output_directory

    except Exception as e:
        traceback.print_exc()

    print("Finished experiment {}, run {}".format(input[0], input[1]))

def run_experiment(number, run, alpha, epsilon):
    
    base_path = os.path.abspath("./build")
    executable_file = os.path.join(base_path, "src", "frozen_lake")
    input_file = os.path.join(base_path, "maps", "frozen_lake_8x8.json")
    
    if not os.path.isfile(executable_file):
        print("No file at path {}".format(executable_file))
        return None

    if not os.path.isfile(input_file):
        print("No file at path {}".format(input_file))
        return None

    output_directory_base = os.path.join(base_path, "ouputs", "exp_{:02}".format(number))

    output_directory = os.path.join(output_directory_base, "run_{:02}".format(run))
    if not os.path.exists(output_directory):
        try:
            os.makedirs(output_directory)
        except OSError:
            print ("Creation of the directory {} failed".format(output_directory))
            return None
    
    p = subprocess.Popen([executable_file, "--agent", "q", "--map", input_file, "--epsilon", str(epsilon),
        "--alpha", str(alpha), "--iteration", str(NUM_ITERATIONS), "--gamma", str(0.99)], 
        cwd=output_directory, stdout=subprocess.DEVNULL)
    p.wait()
    
    return output_directory


def main(num_proc, tasks, start_index=0, start_task_index=0):

    base_path = os.path.abspath("./build")

    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, 0.8, box.height])

    for task_num, task in enumerate(tasks, start=start_task_index):
        experiments = task[1]

        manager = Manager()
        return_dict = manager.dict()

        all_experiments = []
        for i, test in enumerate(experiments, start=start_index):
            for x in range(NUM_RUNS):
                all_experiments.append((i, x, test, return_dict))

        pool = Pool(processes=num_proc)

        pool.imap_unordered(generate_graph, all_experiments)
        pool.close()
        pool.join()

        totals = [[0]*NUM_ITERATIONS for x in range(len(experiments))]
        for key, directory in return_dict.items():
            values = []
            with open(os.path.join(directory, "result.csv")) as f:
                csv_reader = csv.reader(f)
                first = True
                for i, row in enumerate(csv_reader):
                    if first:
                        first = False
                        continue
                    totals[key[0]-start_index][i-1] += float(row[1])

        avg_values = [[x/float(NUM_RUNS) for x in exp_total] for exp_total in totals]

        for i, vals in enumerate(avg_values, start=start_index):
            output_directory_base = os.path.join(base_path, "ouputs", "exp_{:02}".format(i))
            output_path = os.path.join(output_directory_base, 'result_avg.csv')
            print("Writing average for {} to {}".format(i, output_path))
            with open(output_path, '+w') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["Episode", "Reward"])
                for j, x in enumerate(vals):
                    csv_writer.writerow([j, x])

            plt.cla()
            ax.set_position([box.x0, box.y0, 0.8, box.height])
            plt.title(experiments[i-start_index][3])
            plt.xlabel('Iteration')
            plt.ylabel('Reward')
            plt.grid(True)
            plt.plot(vals, label=experiments[i-start_index][3])

            base_path = os.path.abspath("./build")
            ouput_file = os.path.join(base_path, "ouputs", "results_{:02}.png".format(i))
            print("Saving plot to {}".format(ouput_file))
            plt.savefig(ouput_file)

        plt.cla()
        ax.set_position([box.x0, box.y0, 0.625, box.height])
        ax.set_title(task[0])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Reward')
        ax.grid(True)
        for experiment, data in zip(experiments, avg_values):
            ax.plot(data, label=experiment[3])

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        base_path = os.path.abspath("./build")
        ouput_file = os.path.join(base_path, "ouputs", "task_{:02}.png".format(task_num+1))
        print("Saving plot full to {}".format(ouput_file))
        plt.savefig(ouput_file)

        start_index += len(experiments)


    base_path = "./build"
    executable_file = os.path.join(base_path, "src", "frozen_lake")
    input_file = os.path.join(base_path, "maps", "frozen_lake_8x8.json")

    print("\nCommands for running value iteration:")

    for task in tasks:
        for experiment in task[1]:
            if experiment[0]:
                print("{} --agent v --map {} --epsilon {} --alpha {} --iteration {} --gamma 0.99"\
                    .format(executable_file, input_file, experiment[2], experiment[1], NUM_ITERATIONS))


if __name__ == "__main__":

    if len(sys.argv) <= 1:
        num_proc = 1
    else:
        num_proc = sys.argv[1]

    if num_proc is None:
        num_proc = 1
    else:
        num_proc = int(num_proc)

    main(num_proc, pt1_tasks)