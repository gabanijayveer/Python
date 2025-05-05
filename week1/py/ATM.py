import json
import os
# Function to load user data from a JSON file
def load_users(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return json.load(file)
    return {}
# Function to save user data to a JSON file
def save_users(filename, users):
    with open(filename, 'w') as file:
        json.dump(users, file, indent=4)
# Load users from JSON file
filename = 'users.json'
users = load_users(filename)

def check_balance(username, user_pin):
    if username in users and users[username]['pin'] == user_pin:
        return users[username]['balance']
    else:
        return "Invalid username or PIN"
def deposit(username, user_pin, amount):
    if username in users and users[username]['pin'] == user_pin:
        users[username]['balance'] += amount
        save_users(filename, users)  # Save changes to the file
        return f"Deposit successful. New balance is: {users[username]['balance']}"
    else:
        return "Invalid username or PIN"
def withdraw(username, user_pin, amount):
    if username in users and users[username]['pin'] == user_pin:
        if amount > users[username]['balance']:
            return "Insufficient balance."
        users[username]['balance'] -= amount
        save_users(filename, users)  # Save changes to the file
        return f"Withdrawal successful. New balance is: {users[username]['balance']}"
    else:
        return "Invalid username or PIN"
def change_pin(username, old_pin, new_pin):
    if username in users and users[username]['pin'] == old_pin:
        users[username]['pin'] = new_pin
        save_users(filename, users)  # Save changes to the file
        return "PIN changed successfully"
    else:
        return "Invalid username or PIN"
# Main loop
while True:
    print("\n1. Check Balance")
    print("2. Deposit Money")
    print("3. Withdraw Money")
    print("4. Change PIN")
    print("5. Exit")
    choice = input("Choose an option: ")
    username = input("Enter your username: ").capitalize # Get username for each operation
    user_pin = int(input("Enter your PIN: "))  # Get PIN for each operation
    if choice == "1":
        print(check_balance(username, user_pin))
    elif choice == "2":
        amount = float(input("Enter amount to deposit: "))
        print(deposit(username, user_pin, amount))
    elif choice == "3":
        amount = float(input("Enter amount to withdraw: "))
        print(withdraw(username, user_pin, amount))
    elif choice == "4":
        old_pin = int(input("Enter your old PIN: "))
        new_pin = int(input("Enter your new PIN: "))
        print(change_pin(username, old_pin, new_pin))
    elif choice == "5":
        print("Thank you for using our ATM. Goodbye!")
        break
    else:
        print("Invalid option. Please try again.")
# Save the user data to JSON file when the program ends
save_users(filename, users)
