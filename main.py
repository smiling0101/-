from predict import do_predict
from tkinter import *
root=Tk()
def leftClick(event):  # 翻译按钮事件函数
    en_str = Entry1.get()  # 获取要翻译的内容
    print("汉语：",en_str)
    vText = do_predict([en_str])[0]#translate_Word(en_str)
    # vText = do_predict([input("汉语：")])[0]#translate_Word(en_str)
    print("韩语：",vText)
    s.set("")
    Entry2.insert(0, vText)

root.title("汉韩语言通")
root.geometry("700x500")
Label(root, text='输入中文：', width=15,font=('微软雅黑','12','bold')).place(relx=0.3, rely=0.1)  # 绝对坐标（1，1）
Entry1 = Entry(root, width=40)
Entry1.place(relx=0.5, rely=0.1)  # 绝对坐标（110，1）
Label(root, text='输出韩文：', width=15,font=('微软雅黑','12','bold')).place(relx=0.3, rely=0.2)  # 绝对坐标（1，20）
s = StringVar()  # 一个StringVar()对象
s.set("")
Entry2 = Entry(root, width=40, textvariable=s)
Entry2.place(relx=0.5, rely=0.2)  # 绝对坐标（110，20）

Button1 = Button(root, text='翻译', width=8)
Button1.place(relx=0.5, rely=0.3)  # 绝对坐标（40，80）
# 给Label绑定鼠标监听事件
Button1.bind("<Button-1>", leftClick)  # 翻译按钮
root.mainloop()

