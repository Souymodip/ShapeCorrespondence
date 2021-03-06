from tkinter import *
from tkinter import filedialog
import Run
# from tkmacosx import Button


bg_color = 'powderblue'

root = Tk()
root.title("Shape Correspondence")
root.geometry("700x200")
root.focus_set()
root.configure(background=bg_color)


def create_button(parent, text, command):
    return Button(parent, text=text,  command=command, bg='steelblue',fg='white', activebackground='midnightblue',
                  activeforeground=bg_color, highlightbackground=bg_color) # , borderless=1


def create_radio(parent, text, value, command):
    return Radiobutton(parent, text=text, variable=var, value=value,
                  command=command, activebackground=bg_color, highlightbackground=bg_color)


def create_label(parent):
    return Label(parent, bg=bg_color)


def browsefunc1():
    filename = filedialog.askopenfilename()
    label1.config(text=filename, fg="blue")


def browsefunc2():
    filename = filedialog.askopenfilename()
    label2.config(text=filename, fg="blue")


def matchfunc():
    Run.execute(label1.cget("text"), label2.cget("text"), var.get())
    root.update()
    # root.destroy()


button1 = create_button(root, "Browse svg file", browsefunc1)
button1.place(x=10, y=30)
# button1.pack()

button2 = create_button(root, "Browse svg file", browsefunc2)
button2.place(x=10, y=70)
# button2.pack()

button3 = create_button(root, text="Match", command=matchfunc)
button3.place(x=10, y=110)


label1 = create_label(root)
label1.place(x=150, y=35)

# pathlabel1.pack()

label2 = create_label(root)
label2.place(x=150, y=70)
# pathlabel2.pack()

def sel():
   selection = "You selected the option " + str(var.get())
   print(selection)

var = IntVar()
R1 = create_radio(root, text="DFT", value=0, command=sel)
R1.place(x=150, y = 110)

R2 = create_radio(root, text="Procrustes", value=1, command=sel)
R2.place(x=300, y=110)

root.mainloop()