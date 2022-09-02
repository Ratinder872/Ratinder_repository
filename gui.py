from tkinter import *
import tkinter as tk
root = Tk()

root.geometry("655x333")

def name():
    print("RP")
    
#button1.place(x=25, y=100)
 
def age():
    print(20)
    
    
def batch():
    print("CSE")
    
    
def sem():
    print("fifth")
    
    
def college():
    print("miet")
    
    
def city():
    print("jammu")
    
    
def country():
    print("India")
    
    
frame =Frame(root,borderwidth=6,bg ="pink",relief=SUNKEN)
frame.pack(side=LEFT,anchor="n")

b1= Button(frame,fg="red",text="name", command=name)
b1.pack(side=LEFT,padx=22)


b2= Button(frame,fg="blue",text="age",command=age)
b2.pack(side=LEFT,padx=22)

b3= Button(frame,fg="green",text="batch",command=batch)
b3.pack(side=LEFT,padx=22)

b4= Button(frame,fg="black",text="sem",command=sem)
b4.pack(side=LEFT,padx=22)

b5= Button(frame,fg="blue",text="college",command=college)
b5.pack(side=LEFT,padx=22)

b6= Button(frame,fg="red",text="city",command=city)
b6.pack(side=LEFT,padx=22)

b7= Button(frame,fg="red",text="country",command=country)
b7.pack(side=LEFT,padx=22)

root.mainloop()
