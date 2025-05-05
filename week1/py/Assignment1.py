
# Build a calculator that can perform addition, subtraction, multiplication, and division.
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    return a / b

def calculator():
    while True:
    
        choice1 = input("For exit, enter 0 or start again 1: ")
        
        if choice1 == '0':
            print("Exiting the calculator.")
            break
        
        num1 = float(input("Enter first number: "))
        num2 = float(input("Enter second number: "))
        
       
        print("Select operation:")
        print("+. Add")
        print("-. Subtract")
        print("*. Multiply")
        print("/. Divide")
        print("0. Exit")
       
        choice = input("Enter choice (+/-/*/ /0): ")

        
        if choice == '+':
            print(f"Result: {add(num1, num2)}")
        elif choice == '-':
            print(f"Result: {subtract(num1, num2)}")
        elif choice == '*':
            print(f"Result: {multiply(num1, num2)}")
        elif choice == '/':
            if num2 != 0:
                print(f"Result: {divide(num1, num2)}")
            else:
                print("Error! Division by zero.")
        elif choice == '0':
            print("Exiting the calculator.")
            break
        else:
            print("Invalid input. Please select a valid option.")


calculator()
