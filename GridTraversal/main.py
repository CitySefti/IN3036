from tkinter import *
from subprocess import call

def click():
    call(["python", "traverse.py"])

window = Tk(className="Task 1")
window.geometry("300x300")
window.configure(bg="black")

button = Button(window, text="Traverse", command=click, font="Arial", fg="black", bg="white", activeforeground="white", activebackground="white")
button.place(x=150, y = 125)
button.pack()

# idea is to add other buttons for other traverse options (e.g with obstacles)
window.mainloop()

