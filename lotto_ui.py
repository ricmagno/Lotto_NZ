#!/usr/bin/env python3
# Following the tutorial
# https://likegeeks.com/python-gui-examples-tkinter-tutorial/

from tkinter import *
from tkinter.ttk import *
from load_data import *
from constants import *

def last_result(tab):
    df = load_data(file_name)
    t = Text(tab, height=2, width=80)
    t.grid(column=col,row=rows, columnspan=10, rowspan=2)
    t.insert(END, df.tail(1))
    lbl2A = Label(tab, text= 'Last draw was:')
    lbl2A.grid(column=0, row=rows+1)
    return df

window = Tk()

window.title("Lotto prediction LSNN")
window.geometry('870x200')

tab_control = Notebook(window)
tab1 = Frame(tab_control)
tab2 = Frame(tab_control)
tab3 = Frame(tab_control)
tab_control.add(tab1, text='Predict')
tab_control.add(tab2, text='Update')
tab_control.add(tab3, text='Train')
tab_control.pack(expand=1, fill='both')


col = 1
rows = 3

### TAB 1
df = last_result(tab1)


### TAB 2
df = last_result(tab2)

lbl2B = Label(tab2, text= 'Enter:')
lbl2B.grid(column=0, row=rows+31)

# lblb1 = Label(tab2, text= df.tail(1))
# lblb1.grid(column=0, row=2)

lblb1 = Label(tab2, text= 'Draw')
lblb1.grid(column=col, row=rows+30)
splb1 = Spinbox(tab2, from_=0, to=40, width=4)
splb1.grid(column=col,row=rows+31)

lblb2 = Label(tab2, text= 'Date')
lblb2.grid(column=col+1, row=rows+30)
splb2 = Spinbox(tab2, from_=0, to=40, width=10)
splb2.grid(column=col+1,row=rows+31)


b = []
j = 1
for i in balls:
    j= j+1
    lblb3 = Label(tab2, text= i)
    lblb3.grid(column=col+1+j, row=rows+30)
    temp = Spinbox(tab2, from_=0, to=40, width=3)
    b.append(temp)

j = 1
for i in b:
    j= j+1
    i.grid(column=col+1+j,row=rows+31)

def clicked_update():
    ball_draw = [splb1.get(),splb2.get()]
    for i in b:
        ball_draw.append(i.get())

    df = update(ball_draw)
    print(df.tail(2))
    save(df)



btn = Button(tab2, text="Update", command=clicked_update)
btn.grid(column=12, row=34)


lbl3 = Label(tab3, text= 'How many features?')
lbl3.grid(column=0, row=0)

lbl3 = Label(tab3, text= 'Is the data sorted?')
lbl3.grid(column=1, row=0)
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


def clicked_train():
    print('Nothing happens here.')

    # res = "Welcome to " + txt.get()
    # lbl.configure(text= res)

btn = Button(tab3, text="Train", command=clicked_train)
btn.grid(column=2, row=5)
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
