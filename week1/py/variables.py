# Basic Python Syntax
a = 10
b = 20.5
name = "ABC"
is_active = True

sum = a + b
difference = a - b
product = a * b


print("Sum:", sum)
print("Difference:", difference)
print("Product:", product)


number = 10
if number > 0:
    print("Positive")
else:
    print("Non-positive")

for i in range(5):
    print(i)

#Control Structures: 
count = 0
while count < 5:
    if count == 3:
        break  
    elif count == 1:
        count += 1
        continue 
    print(count)
    count += 1

#def function

def greet(name, greeting="Hello"):
    print(f"{greeting}, {name}!")

greet("shinees")
greet("jennifer", greeting="Good morning")


class Family:
    def __init__(self, count):
        self.count = count
    
    def member(self):
        print(f"{self.count} member is there")


members = Family("5")
members.member() 

class Family:
    def __init__(self, name):
        self.name = name
    
    def member(self):
        print(f"{self.name} member there")

class Function(Family):
    def attend(self):
        print(f"{self.name} ")

farm = Function("Buddy")
farm.member()
farm.attend() 

def sum():
    x = 10  
    print(x) 

sum()

 # Global variable

x = 12

def sum():
    print(x)  

sum() 
print(x) 


 # Global variable out side
x = 10 

def sums():
    x = 20 # same name inside
    print("function:", x)  

sums()
print("Outside function:", x)  


try:
    x = 10 / 0 
except ZeroDivisionError as e:
    print(f"Error: {e}")
if(10>0):
	
		sum = x + 2
		print(sum)
else:
    print("No errors occurred.")

# ___________________

try:
    x = 10 / 0 
except ZeroDivisionError as e:
    print(f"Error: {e}")

else:
    print("No errors occurred.")
finally:
    print("This always runs.")

# File Handling: Reading/writing text and binary files
with open("sample.txt", "w") as f:
    f.write("Hello, World!")

# Reading from a text file
with open("sample.txt", "r") as f:
    content = f.read()
    print(content)

students = {
    "Alice": 85,
    "Bob": 72,
    "Charlie": 90,
    "David": 65,
    "Eva": 50
}

for name, marks in students.items():
    if marks >= 90:
        grade = "A+"
    elif marks >= 80:
        grade = "A"
    elif marks >= 70:
        grade = "B"
    elif marks >= 60:
        grade = "C"
    elif marks >= 50:
        grade = "D"
    else:
        grade = "F"
    
    print(f"{name} scored {marks} and received grade {grade}")

numbers= [10, -5, 25, 0, -8, 17, 50, -1, 99]

positive_count = 0
total = 0

for num in numbers:
    if num < 0:
        print(f"Skipping negative number: {num}")
        continue  # skip negatives

    if num == 0:
        print("Zero found. Ending loop.")
        break  # stop loop on zero

    total += num
    positive_count += 1

print(f"Count of positive numbers: {positive_count}")
print(f"Total sum of positive numbers: {total}")
