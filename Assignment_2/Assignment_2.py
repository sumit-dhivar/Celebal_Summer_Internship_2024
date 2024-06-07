# -*- coding: utf-8 -*-
"""
Created on Wed May 29 18:37:35 2024

@author: Sumit Dhivar
"""

def take_input():# function to take the input operands
    num1 = int(input("Enter the first operand : "))
    num2 = int(input("Enter the second operand : "))
    return num1 , num2
    
def Add(num1 , num2):# function to perform addition
    res = num1 + num2
    return res

def Sub(num1 , num2):# function to perform Substraction
    res = num1 - num2
    return res
    
def Mul(num1 , num2):# function to perform Multiplication
    res = num1 * num2
    return res

def Div(num1 , num2):# function to perform Division
    res = num1 / num2
    return res

def f_div(num1 , num2):# function to perform floar Division 
# floar division always gives output as an integer value
    res = num1 // num2
    return res
    
def power(num1 , num2):# function to perform raised to operation
    res = num1**num2
    return res

print("-----Welcome to the Calculator-----")
while(True):# creating a menu for the user to decide which operation 
# he/she wants to perform
    print("Select a choice from the Below Menu")
    print("1 - Addition")
    print("2 - Substraction")
    print("3 - Multiplication")
    print("4 - Division")
    print("5 - floar Division")
    print("6 - Power operation")
    print("7 - Exit from the Calculator")
    
    ch = int(input("Provide an option : "))
    # storing the user's choice in one variable
    # accoring to users choice the if-statement is executed
    
    if(ch == 1):
        num1 , num2 = take_input() # calling take_input() to take the input from the user
        res = Add(num1 , num2) # Calling the Add() to Perform addition
        print(f"Addition of {num1}+{num2} is {res}\n")
        
    if(ch == 2):
        num1 , num2 = take_input()
        res = Sub(num1 , num2)
        print(f"Substration of {num1}-{num2} is {res}\n")
    
    if(ch == 3):
        num1 , num2 = take_input()
        res = Mul(num1 , num2)
        print(f"Multiplication of {num1}*{num2} is {res}\n")
    
    if(ch == 4):
        num1 , num2 = take_input()
        res = Div(num1 , num2)
        print(f"Substration of {num1}/{num2} is {res}\n")
        
    if(ch == 5):
        num1 , num2 = take_input()
        res = f_div(num1 , num2)
        print(f"Floar Division of {num1}//{num2} is {res}\n")
   
    if(ch == 6):
        num1 , num2 = take_input()
        res = power(num1 , num2)
        print(f"Power of {num1}^{num2} is {res}\n")    
    
    if(ch == 7):
        break # using the break statement the infinte loop will get break 
    
    
    
    
    