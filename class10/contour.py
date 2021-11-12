import csv
import numpy as np
import matplotlib.pyplot as plt

def read_csv(filename):
    Z = []
    with open(filename) as fp:
        reader = csv.reader(fp)
        for row in reader:
            row = list(map(float, row))
            Z.append(row)
    return Z

def main():
    X = np.linspace(0, 1, 32)
    Y = np.linspace(0, 1, 32)
    Z = read_csv("data.csv")
    print(Z)

    fig, ax=plt.subplots(1,1)
    cp = ax.contourf(X, Y, Z)
    fig.colorbar(cp) # Add a colorbar to a plot
    ax.set_title('Filled Contours Plot')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    plt.savefig("image.png")
    # plt.show()

if __name__ == "__main__": main()
