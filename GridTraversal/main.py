from tkinter import *
from subprocess import call


def click():
    call(["python", "traverse.py"])


window = Tk(className="Task 1")
window.geometry("300x300")
window.configure(bg="black")

button = Button(window, text="Traverse", command=click, font="Arial", fg="black", bg="white", activeforeground="white",
                activebackground="white")
button.place(relx=0.5, rely=0.5, anchor=CENTER)

# idea is to add other buttons for other traverse options (e.g. with obstacles)
window.mainloop()
