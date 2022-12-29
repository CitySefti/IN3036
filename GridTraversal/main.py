from tkinter import *
from subprocess import call
from traverse import Traverse

"""
def click():
    call(["python", "traverse.py"])
"""
window = Tk(className="Task 1")
window.geometry("350x350")

minLabel = Label(window, text="Minimum")
minLabel.place(x=80, y=50)
cellMinInput = Entry(window)
cellMinInput.place(x=180, y=50)

maxLabel = Label(window, text="Maximum")
maxLabel.place(x=80, y=100)
cellMaxInput = Entry(window)
cellMaxInput.place(x=180, y=100)

widthLabel = Label(window, text="Width")
widthLabel.place(x=80, y=150)
widthInput = Entry(window)
widthInput.place(x=180, y=150)

heightLabel = Label(window, text="Height")
heightLabel.place(x=80, y=200)
heightInput = Entry(window)
heightInput.place(x=180, y=200)


def click():
    if cellMinInput.get() != 0:
        min = int(cellMinInput.get())

    if cellMaxInput.get() != 0:
        max = int(cellMaxInput.get())

    if cellMinInput.get() != 0:
        width = int(widthInput.get())

    if cellMinInput.get() != 0:
        height = int(heightInput.get())

    newAgent = Traverse(cellMin=min, cellMax=max, gridWidth=width, gridHeight=height)
    newAgent.run()

button = Button(window, text="Traverse", command=click, font="Arial")
button.place(x=130, y=250)

# idea is to add other buttons for other traverse options (e.g. with obstacles)
window.mainloop()

#hello = Traverse(cellMin=0, cellMax=10, gridWidth=10, gridHeight=10)
#hello.run()
