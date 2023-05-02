from IPython import display
import matplotlib.pyplot as plt

plt.ion()   # Enable interactive mode


def plot_hit(hitself, hitboundery):
    fig = plt.figure(num=1)
    display.clear_output(wait=True)
    display.display(plt.gcf(), display_id=True)     # gcf - Get the current figure.
    plt.clf()                                       # Clear the current figure
    plt.title('Training...')
    plt.xlabel("Number of games")
    plt.ylabel("Accumulated counts")
    plt.plot(hitself, label="Hit itself")
    plt.plot(hitboundery, label='Hit_boundary')
    plt.ylim(ymin=0)
    plt.legend()
    plt.text(len(hitself) - 1, hitself[-1], str(hitself[-1]))
    plt.text(len(hitboundery) - 1, hitboundery[-1], str(hitboundery[-1]))
    plt.show(block=False)   # Ensure that all figure windows are displayed and return immediately
    plt.pause(.1)           # Run the GUI event loop for interval seconds.
