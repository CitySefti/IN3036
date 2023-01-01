from tkinter import *
from traverse import Traverse
from traversepq import TraversePQ

# Create frame/window
window = Tk(className="Task 1")
window.geometry("350x350")

# Make Labels and input boxes for parameters
minLabel = Label(window, text="Minimum")
minLabel.place(x=80, y=50)
minInput = Entry(window)
minInput.place(x=180, y=50)

maxLabel = Label(window, text="Maximum")
maxLabel.place(x=80, y=100)
maxInput = Entry(window)
maxInput.place(x=180, y=100)

widthLabel = Label(window, text="Width")
widthLabel.place(x=80, y=150)
widthInput = Entry(window)
widthInput.place(x=180, y=150)

heightLabel = Label(window, text="Height")
heightLabel.place(x=80, y=200)
heightInput = Entry(window)
heightInput.place(x=180, y=200)


# Implements button functionality
def click():
    # Enough checks there to prevent freezing/crashing but no error box
    if minInput.get() != 0:
        if minInput.get().isnumeric():
            min = int(minInput.get())

    if maxInput.get() != 0:
        if maxInput.get().isnumeric():
            max = int(maxInput.get())

    if widthInput.get() != 0:
        if widthInput.get().isnumeric():
            width = int(widthInput.get())

    if heightInput.get() != 0:
        if heightInput.get().isnumeric():
            height = int(heightInput.get())

    #Creates new traversable map with set parameters, runs it, which returns results.
    newAgent = Traverse(cellMin=min, cellMax=max, mapWidth=width, mapHeight=height)
    newAgent.run()

    pqAgent = TraversePQ(cellMin=min, cellMax=max, mapWidth=width, mapHeight=height)
    pqAgent.run()


# Button, the idea here was to add other buttons for other traverse options (e.g. with obstacles)
button = Button(window, text="Traverse", command=click, font="Arial")
button.place(x=130, y=250)

# Have frame/window be visible
window.mainloop()
