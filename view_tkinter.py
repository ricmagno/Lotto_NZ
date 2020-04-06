#!/usr/bin/python

import tkinter as tk
from tkinter import *
import tkinter.messagebox

class App(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()

# create the application
myapp = App()
#
# here are method calls to the window manager class
#
myapp.master.title("My Do-Nothing Application")
myapp.master.geometry("320x200")
myapp.master.maxsize(1000, 400)
myapp.master.configure(background='black')



def helloCallBack():
   tkinter.messagebox.showinfo( "Hello Python", "Hello World")
#
B = tkinter.Button(myapp, text ="Hello", background='black', command = helloCallBack)
B.pack()

# start the program
myapp.mainloop()
#
#
# TK_SILENCE_DEPRECATION=1
#
# def clickExitButton(top):
#     exit()
#
#
# top = tkinter.Tk()
#
# def helloCallBack():
#    tkinter.messagebox.showinfo( "Hello Python", "Hello World")
#
# B = tkinter.Button(top, text ="Hello", command = helloCallBack)
# # exitButton = tkinter.Button(top, text="Exit", command=clickExitButton(top))
#
#
# # exitButton.place(x=0, y=0)
#
# B.pack()
# # exitButton.pack()
#
# top.mainloop()
