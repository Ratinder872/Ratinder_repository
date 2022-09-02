  x = 2     # global variable [has a property by which we can declear it inside

# as well as outside the function]

def myfunc():
   
    a="MEHUL"
    print(a)
    
    print(x) # printing global variable inside the function


myfunc()
print("global variable:",x)  # printing outside the func
