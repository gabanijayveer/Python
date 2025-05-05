# Practice basic OOP by creating a class for Person with attributes like name, age, etc.

class Person:
   
    def __init__(self):
        self.name = input("Enter name: ")
        self.age = float(input("Enter age: "))  
        self.gender = input("Enter gender: ")

    def display_info(self):
        print(f"Name: {self.name}")
        print(f"Age: {self.age}")
        print(f"Gender: {self.gender}")

    def currentage(self):
        self.age 
        print(f"Age, {self.name}! You are now {self.age} years old.")

    def introduce(self):
        print(f"Hi, my name is {self.name}. I am {self.age} years old and I am {self.gender}.")


person1 = Person()

person1.display_info()
person1.introduce()
person1.currentage() 


person2 = Person()

person2.display_info()
person2.introduce()
