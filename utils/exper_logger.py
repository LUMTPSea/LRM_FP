import os
import csv
import matplotlib.pyplot as plt

class Logger(object):
    """
    Logger saves the labels, legend and paths of the plot and log file
    """
    def __init__(self, log_dir):
        """
        Args:
            log_dir: the path of the log files
        """
        self.log_dir = log_dir
        self.txt_path = os.path.join(log_dir, 'log.txt')
        self.csv_path = os.path.join(log_dir, 'performance.csv')
        self.fig_path = os.path.join(log_dir, 'fig.png')

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.txt_file = open(self.txt_path, 'w')
        self.csv_file = open(self.csv_path, 'w')
        fieldnames = ['iteration', 'nash_conv']
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

    def log(self, text):
        self.txt_file.write(text + '\n')
        self.txt_file.flush()
        # print(text)

    def log_performance(self, iteration, nash_conv, value):
        """
        Args:
            iteration: the iteration of the current point
            nash_conv: the nash_conv of the current point
        Returns:

        """
        self.writer.writerow({'iteration': iteration, 'nash_conv': nash_conv})
        # print('')
        self.log('-------------------')
        self.log('iteration      |' + str(iteration))
        self.log('nash_conv      |' + str(nash_conv))
        self.log('best_responder0 |' + str(value[0]))
        self.log('best_responder1 |' + str(value[1]))

        self.log('-------------------')

    def plot(self, algorithm):
        self.plot_path(self.csv_path, self.fig_path, algorithm)

    def close_files(self):
        """
        Close the created file objects

        """
        if self.txt_path is not None:
            self.txt_file.close()
        if self.csv_path is not None:
            self.csv_file.close()

    def plot_path(self, csv_path, save_path, algorithm):
        """
            Read data from csv file and plot the results

        Args:
            csv_path:
            save_path:
            algorithm:

        Returns:

        """
        with open(csv_path) as csvfile:
            print(csv_path)
            reader = csv.DictReader(csvfile)
            xs = []
            ys = []
            for row in reader:
                xs.append(int(row['iteration']))
                ys.append(float(row['nash_conv']))
            fig, ax = plt.subplots()
            ax.plot(xs, ys, label=algorithm)
            ax.set(xlabel='iteration', ylabel='nash_conv')
            ax.legend()
            ax.grid()

            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            fig.savefig(save_path)
