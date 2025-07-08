# a=input("enter a number A")
# b=input ("enter a number B")
# temp=a
# a=b
# b=temp
# print("A=",a)
# print("B= ",b)

# a=95
# b=89
# c=97
# per=((a+b+c)/300)*100
# print(per)

# print("        1")
# print("    1         1")anyannn
# print("1                   1")

# a=75
# b=80
# c=87
# d=96
# e=92
# avg=a+b+c+d+e/5
# print(avg)

# r=float(input("enter radius"))
# b=3.14*r*r
# print(b)

# a=int(input("enter a number 1"))
# b=int(input("enter a number 2"))
# c=int(input("enter a number 3"))
# if(a>b)and(a>c):
#     print("a is greater")
# elif(b>a)and(b>c):
#     print("b is greater")
# else:
#     print("c is greater")

# import math
# x=float(input("enter a number 1"))
# deg=x*(180/math.pi)
# print(deg)

import math
# x=float(input("enter a number 1"))
# f=(x*(9/5))+32
# print(f)

# =float(input("enter the principle"))
# t=float(input("enter the time"))
# r=float(input("enter the rate"))
# si=(p*t*r)/100
# print(si)p

# for i in range(0,15):
#     for j in range(0,i):
#         print("love you",end="")
#     print("\n")

# a=int(input("enter a:"))
# for i in range(0,6):
#     for j in range(0,i):
#         print(i,end="")
#     print("\n")

# def sheet(roll,mark1,mark2,mark3,mark4,mark5):
#     print("roll number:",roll)
#     print("subject 1:",mark1)
#     print("subject 2:",mark2)
#     print("subject 3:",mark3)
#     print("subject 4:",mark4)
#     print("subject 5:",mark5)
#     print("total:",mark1+mark2+mark3+mark4+mark5)
# sheet(mark1=40,mark2=60,mark3=50,mark4=70,mark5=75,roll=46)

# def fib(n):
#     if n<=0:
#         print("incorrect input")
#     elif n==1:
#         return 1
#     elif n==2:
#         return 2
#     else:
#         return(fib(n-1)+fib(n-2))
# n=int(input("enter a position"))
# print(fib(n))

#  area(b,l=10):
#     return b*ldef
# print(area(10))

# r=int(input("radius"))
# area=lambda r:3.14*r*r
# print(area(r))

# def calculator(a,b):
#     return(a+b,a-b,a*b,a//b)
# n1=int(input("enter a"))
# n2=int(input("enter b"))
# print(calculator(n1,n2))

# def cam(cat1=0,cat2=0,ass=0):
#     return cat1+cat2+ass
# cat1=int(input("cat1:"))
# cat2=int(input("cat2:"))
# ass=int(input("assignment:"))
# print(cam(cat1,cat2,ass))

# def sum(n):
#     if n<10:
#         return n
#     else:
#         return n%10+sum(n//10)
# num=int(input("enter a number:"))
# digit=sum(num)
# print("sum of digit:",digit)

# def convert(s):
#     s=s.lower()
#     s=s.replace(".","").replace(",","")
#     return s
# s=input("enter a string")
# print(convert(s))

# s= input("enter intput")
# f=s[0]
# sec=s[1:]
# h=f+sec.replace(f,"$")
# print(h)

# s=input("enter string")
# char=s[0]
# rev=s[1:]
# s=char+rev.replace(char,"$")
# print(s)

# s=input("string")
# print(s.swapcase())

# def printvalue(s1,s2):
#     s1=str(s1)
#     s2=str(s2)
#     if len(s1)>len(s2):
#         print(s1)
#     elif len(s2)>len(s1):
#         print(s2)
#     else:
#         print(s1,s2)
# s1=input("string1:")
# s2=input("string2:")
# printvalue(s1,s2)

# password="preethi"
# for i in range(0,3):
#     pas=input("enter password")
#     if pas==password:
#         print("welcome")
#         break
#     else:
#         print("try again")
#         continue

# s=input("enter the string")
# a= s.replace(" ","")
# print(a)

# s=input("enter string")
# a=s.replace(" ","")
# print(a)

# s=input("enter string")
# num,alpha,spl=0,0,0
# for i in range(len(s)):
#     if s[i].isalpha():
#         alpha+=1
#     elif s[i].isdigit():
#         num+=1
#     else:
#         spl+=1
# print("number=",num)
# print("alphabet=",alpha)
# print("special=",spl)

# def bmi(wei,hei):
#     bmi=wei/hei**2
#     return bmi
# wei=int(input("weight in kg"))
# hei=int(input("height in cm"))
# print (bmi(wei,hei))

# a=input("enter string")
# for i in range(len(a)):
#     print(a[i]*(i+1))

# a=0
# b=1
# c=a+b
# print(a,b,c,end="")
# for i in range(3,11):
#     if c>10:
#         break
#     else:
#         a=b
#         b=c
#         c=a+b
#         if(c!=2):
#             print(c,end=" "  area(b,l=10):
