#%%
age = 20
name  = 'limeihang'
print('{0} was {1} years old '.format(name,age))

# %%
print('{} was {} years old when he wrote this book'.format(name,age))
# %%
x = 50
def func():
    global x
    print('x is ', x)
    x= 2
    print('changed x to : {}'.format(x))

func()
print('value of x is {}'.format(x))
# %%
import sys
print('the command line arguments are:')
for i in sys.argv:
    print(i)


# %%
if  __name__ =='__main__':
    print('this program is being run by itself')
else:
    print('i am being imported from another module')