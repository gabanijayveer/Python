import json

class Item:
    def __init__(self, name, quantity, price):
        self.name = name
        self.quantity = quantity
        self.price = price
    def total_value(self):
        return self.quantity * self.price
    def __str__(self):
        return f"{self.name}: {self.quantity} units at ${self.price:.2f} each"

class Inventory:
    def __init__(self):
        self.items = {}
    def add_item(self, name, quantity, price):
        if name in self.items:
            self.items[name].quantity += quantity
        else:
            self.items[name] = Item(name, quantity, price)
        print(f"Added {quantity} of {name} to inventory.")
    def view_items(self):
        if not self.items:
            print("Inventory is empty.")
            return
        for item in self.items.values():
            print(item)
    def save_to_file(self, filename):
        with open(filename, 'w') as file:
            json.dump({name: vars(item) for name, item in self.items.items()}, file)
        print(f"Inventory saved to {filename}.")
    def load_from_file(self, filename):
        try:
            with open(filename, 'r') as file:
                data = json.load(file)
                for name, item_data in data.items():
                    self.items[name] = Item(name, item_data['quantity'], item_data['price'])
            print(f"Inventory loaded from {filename}.")
        except FileNotFoundError:
            print(f"Error: The file {filename} does not exist.")
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON from the file.")
# Define a function to display the menu
def display_menu():
    print("\nInventory Management System")
    print("1. Add Item")
    print("2. View Items")
    print("3. Save Inventory")
    print("4. Load Inventory")
    print("5. Exit")
# Main program loop
def main():
    inventory = Inventory()
    while True:
        display_menu()
        choice = input("Choose an option (1-5): ")
        if choice == "1":
            name = input("Enter item name: ")
            try:
                quantity = int(input("Enter item quantity: "))
                price = float(input("Enter item price: "))
                inventory.add_item(name, quantity, price)
            except ValueError:
                print("Error: Quantity must be an integer and price must be a float.")
        elif choice == "2":
            inventory.view_items()
        elif choice == "3":
            filename = input("Enter filename to save inventory: ")
            inventory.save_to_file(filename)
        elif choice == "4":
            filename = input("Enter filename to load inventory: ")
            inventory.load_from_file(filename)
        elif choice == "5":
            print("Exiting the inventory management system.")
            break
        else:
            print("Invalid option. Please try again.")
if __name__ == "__main__":
    main()
