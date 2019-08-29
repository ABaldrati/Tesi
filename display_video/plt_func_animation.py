import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


# Funzione che genera un video con le predizioni precedentemente generate
def create_prediction_video(path_prediction, path_videostats_prediction):
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    matrix = np.load(path_prediction)
    row = matrix[0]
    frames = len(matrix)
    x = np.arange(4)
    barcollection = plt.bar(x, height=[row[0], row[1], row[2], row[3]], color=['orange', 'blue', 'yellow', 'black'])
    plt.xticks(x, ['back', 'card', 'subs', 'goal'])
    plt.ylabel('Probability')
    axes = plt.gca()
    axes.set_ylim([0, 1])

    def animate(i):
        row = matrix[i]
        ax.set_title('{}:{}'.format(i // 60, i % 60))
        for i, b in enumerate(barcollection):
            b.set_height(row[i])

    anim = animation.FuncAnimation(fig, animate, repeat=False, blit=False, frames=frames)

    anim.save(path_videostats_prediction, writer=animation.FFMpegWriter(fps=1))
    plt.close(fig)
