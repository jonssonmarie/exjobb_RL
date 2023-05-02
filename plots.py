from IPython import display
import matplotlib.pyplot as plt

plt.ion()       # Enable interactive mode


def plot(scores, mean_scores):
    fig = plt.figure(num=2)
    display.clear_output(wait=True)
    display.display(plt.gcf(), display_id=True)     # gcf - Get the current figure.
    plt.clf()                                       # Clear the current figure
    plt.title('Training...')
    plt.xlabel("Number of games")
    plt.ylabel("Score")
    plt.plot(scores, label="scores")
    plt.plot(mean_scores, label='mean scores')
    plt.ylim(ymin=0)
    plt.legend()
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)       # Ensure that all figure windows are displayed and return immediately
    plt.pause(.1)               # Run the GUI event loop for interval seconds.
