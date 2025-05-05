
shopping_cart = {}

items_catalog = {
    "Apple": 50,     
    "Banana": 30,
    "Orange": 80,
    "Milk": 60,
    "Bread": 45,
    "Eggs": 20,
}

def view_cart():
    if shopping_cart:
        print("\nYour Shopping Cart:")
        total_price = 0
        for item, quantity in shopping_cart.items():
            price = items_catalog[item]  
            total_price += price * quantity
            print(f"{item}: {quantity} x ₹{price} = ₹{price * quantity}")
        print(f"Total: ₹{total_price:.2f}")
    else:
        print("\nYour cart is empty!")

def add_to_cart():
    item = input("\nEnter the item you want to add: ").capitalize()
    if item in items_catalog:
        quantity = int(input(f"How many {item}s would you like to add? "))
        if item in shopping_cart:
            shopping_cart[item] += quantity  
        else:
            shopping_cart[item] = quantity  
        print(f"Added {quantity} {item}(s) to your cart.")
    else:
        print("Item not available in the catalog!")

def remove_from_cart():
    item = input("\nEnter the item you want to remove: ").capitalize()
    if item in shopping_cart:
        quantity = int(input(f"How many {item}s would you like to remove? "))
        if quantity >= shopping_cart[item]:
            del shopping_cart[item]  
            print(f"Removed {item} from the cart.")
        else:
            shopping_cart[item] -= quantity  
            print(f"Removed {quantity} {item}(s) from the cart.")
    else:
        print(f"{item} is not in your cart!")

def show_menu():
    while True:
        print("\n=== Shopping Cart Menu ===")
        print("1. View Cart")
        print("2. Add Item to Cart")
        print("3. Remove Item from Cart")
        print("4. Exit")
        
        choice = input("Choose an option (1/2/3/4): ")
        
        if choice == '1':
            view_cart()
        elif choice == '2':
            add_to_cart()
        elif choice == '3':
            remove_from_cart()
        elif choice == '4':
            print("Thank you for shopping! Goodbye.")
            break
        else:
            print("Invalid option, please try again.")

show_menu()
