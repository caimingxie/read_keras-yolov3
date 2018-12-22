
name = ['ming','zhang','xie']
i=1

# try 预期出现异常的程序

try:
    name[i]
# except 表示异议
except:
    print('异常')
# 无异常状态
else:
    print(name[i],'没有异常')

finally:
    print('this is a demo')


# assert false 时候报错
assert 1==3,'这是一个异常'

