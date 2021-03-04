from tkinter import *
from tkinter import filedialog
import Run

root = Tk()
root.title("Shape Correspondence")
root.geometry("700x200")
root.lift()

files = ""

def browsefunc1():
    filename = filedialog.askopenfilename()
    label1.config(text=filename, fg="blue")

def browsefunc2():
    filename = filedialog.askopenfilename()
    label2.config(text=filename, fg="blue")

def matchfunc():
    Run.execute(label1.cget("text"), label2.cget("text"))
    root.update()
    root.destroy()

button1 = Button(root, text="Browse svg file1", command=browsefunc1)
button1.place(x=10, y=30)
# button1.pack()

button2 = Button(root, text="Browse svg file2", command=browsefunc2)
button2.place(x=10, y=70)
# button2.pack()

button3 = Button(root, text="Match", command=matchfunc)
button3.place(x=10, y=110)


label1 = Label(root)
label1.place(x=150, y=35)

# pathlabel1.pack()

label2 = Label(root)
label2.place(x=150, y=70)
# pathlabel2.pack()


root.mainloop()