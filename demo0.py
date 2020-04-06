#!/usr/bin/env python3
# Following the tutorial
# https://likegeeks.com/python-gui-examples-tkinter-tutorial/

from tkinter import *
from tkinter.ttk import *

balls = ['1', '2', '3', '4', '5', '6', 'Bonus', 'Powerball']



window = Tk()

window.title("Lotto prediction LSNN")
window.geometry('850x200')


tab_control = Notebook(window)
tab1 = Frame(tab_control)
tab2 = Frame(tab_control)
tab3 = Frame(tab_control)
tab_control.add(tab1, text='Predict')
tab_control.add(tab2, text='Update')
tab_control.add(tab3, text='Train')


lbl1 = Label(tab1, text= 'label1')
lbl1.grid(column=0, row=0)


lbl2A = Label(tab2, text= 'Last draw was:')
lbl2A.grid(column=0, row=0)

lbl2B = Label(tab2, text= 'Enter the stuff')
lbl2B.grid(column=0, row=1)
# for i in range(1,9,1):
    # print(i)
lblb1 = Label(tab2, text= '1')
lblb1.grid(column=0, row=2)



col = 1
rows = 3


lblb1 = Label(tab2, text= 'Draw')
lblb1.grid(column=col, row=rows)
splb1 = Spinbox(tab2, from_=0, to=40, width=3)
splb1.grid(column=col,row=rows+1)

lblb2 = Label(tab2, text= 'Date')
lblb2.grid(column=col+1, row=rows)
splb2 = Spinbox(tab2, from_=0, to=40, width=3)
splb2 = Spinbox(tab2, from_=0, to=40, width=3)
splb2.grid(column=col+1,row=rows+1)


b = []
j = 1
for i in balls:
    j= j+1
    lblb3 = Label(tab2, text= i)
    lblb3.grid(column=col+1+j, row=rows)
    temp = Spinbox(tab2, from_=0, to=40, width=3)
    b.append(temp)

j = 1
for i in b:
    j= j+1
    i.grid(column=col+1+j,row=rows+1)


# B1 = Spinbox(tab2, from_=0, to=40, width=3)
# B1.grid(column=col+1,row=rows)
# B2 = Spinbox(tab2, from_=0, to=40, width=3)
# B2.grid(column=col+2,row=rows)
# B3 = Spinbox(tab2, from_=0, to=40, width=3)
# B3.grid(column=col+3,row=rows)
# B4 = Spinbox(tab2, from_=0, to=40, width=3)
# B4.grid(column=col+4,row=rows)
# B5 = Spinbox(tab2, from_=0, to=40, width=3)
# B5.grid(column=col+5,row=rows)
# B6 = Spinbox(tab2, from_=0, to=40, width=3)
# B6.grid(column=col+6,row=rows)
# B7 = Spinbox(tab2, from_=0, to=40, width=3)
# B7.grid(column=col+7,row=rows)
# B8 = Spinbox(tab2, from_=0, to=40, width=3)
# B8.grid(column=col+8,row=rows)
# B9 = Spinbox(tab2, from_=0, to=10, width=3)
# B9.grid(column=col+9,row=rows)



def clicked():
    for i in b:
        print(splb1.get(), splb2.get(), i.get())
        # res = "Welcome to " + txt.get()
        # lbl.configure(text= res)

btn = Button(tab2, text="Update", command=clicked)
btn.grid(column=12, row=4)


tab_control.pack(expand=1, fill='both')


lbl3 = Label(tab3, text= 'label2')
lbl3.grid(column=0, row=0)

# lbl = Label(window, text="Lotto", font=("Arial Bold", 50))
# lbl.grid(column=0, row=0)
#
#
# txt = Entry(window, width=20)
# txt.focus()
# txt.grid(column=1, row=0)
#
#
# # def clicked():
#         # lbl.configure(text="Button was clicked !!")
#
# def clicked():
#     res = "Welcome to " + txt.get()
#     lbl.configure(text= res)
#
#
# # btn = Button(window, text="Click Me", bg="orange", fg="red", command=clicked)
# # btn.grid(column=2, row=0)
#
combo1 = Combobox(tab3)
combo1['values']= ('Univariate', 'Multivariate')
combo1.current(1) #set the selected item
combo1.grid(column=0, row=5)
#
combo2 = Combobox(tab3)
combo2['values']= ('Unsorted', 'Sorted')
combo2.current(1) #set the selected item
combo2.grid(column=1, row=5)

def clicked():
    res = "Welcome to " + txt.get()
    lbl.configure(text= res)

btn = Button(tab3, text="Train", command=clicked)
btn.grid(column=0, row=6)
#
# chk_state = BooleanVar()
# chk_state.set(True) #set check state
# chk = Checkbutton(window, text='Choose', var=chk_state)
# chk.grid(column=0, row=2)
#
#
# rad1 = Radiobutton(tab3,text='Univariate', value=1)
# rad2 = Radiobutton(tab3,text='Multivariate', value=2)
# rad1.grid(column=0, row=3)
# rad2.grid(column=1, row=3)
#
# rad3 = Radiobutton(tab3,text='Sorted', value=3)
# rad4 = Radiobutton(tab3,text='Unorted', value=4)
# rad3.grid(column=0, row=4)
# rad4.grid(column=1, row=4)


window.mainloop()
