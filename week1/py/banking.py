# Initial setup
pin = 1234
balance = 220000

# Main loop
while True:
    print("\n1. Check Balance")
    print("2. Deposit Money")
    print("3. Withdraw Money")
    print("4. Change PIN")
    print("5. Exit")
    choice = input("Choose an option: ")
    user_pin = int(input("Enter your PIN: "))  # Move PIN input here for each operation
    def check_balance(user_pin):
        if user_pin == pin:
            return balance
        else:
            return "Invalid PIN"
    def deposit(user_pin, amount):
        global balance
        if user_pin == pin:
            balance += amount
            return "Deposit successful. New balance is: " + str(balance)
        else:
            return "Invalid PIN"
    def withdraw(user_pin, amount):
        global balance
        if user_pin == pin:
            if amount > balance:
                return "Insufficient balance."
            balance -= amount
            return "Withdrawal successful. New balance is: " + str(balance)
        else:
            return "Invalid PIN"
    def change_pin(old_pin, new_pin):
        global pin
        if old_pin == pin:
            pin = new_pin
            return "PIN changed successfully"
        else:
            return "Invalid PIN"

    if choice == "1":
        print(check_balance(user_pin))
    elif choice == "2":
        amount = float(input("Enter amount to deposit: "))
        print(deposit(user_pin, amount))
    elif choice == "3":
        amount = float(input("Enter amount to withdraw: "))
        print(withdraw(user_pin, amount))
    elif choice == "4":
        old_pin = int(input("Enter your old PIN: "))
        new_pin = int(input("Enter your new PIN: "))
        print(change_pin(old_pin, new_pin))
    elif choice == "5":
        print("Thank you for using our ATM. Goodbye!")
        break
    else:
        print("Invalid option. Please try again.")
